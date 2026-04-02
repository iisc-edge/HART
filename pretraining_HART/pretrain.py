import logging
import math
import os
import tempfile
import hashlib
from pathlib import Path
import torch 
import numpy as np
import pandas as pd
import random
import json
from torch.utils.data import Dataset
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint
import torch.distributed as dist
from torch.utils.data import IterableDataset

from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction
from tsfm_public.models.tspulse import TSPulseConfig
from tsfm_public.models.tinytimemixer.utils import get_ttm_args
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.visualization import plot_predictions

logger = logging.getLogger(__name__)


import random
import itertools
from pathlib import Path
from torch.utils.data import IterableDataset
import pandas as pd
import numpy as np
import random
import itertools
from pathlib import Path
from torch.utils.data import IterableDataset

class StreamingTimeSeriesDataset(IterableDataset):
    # Columns that identify metadata — never treated as sensor signals
    META_COLS = {"subject_id", "activity_id", "label", "activity", "subject", "session_id"}

    def __init__(
        self,
        data_root_path: str,
        context_length: int,
        prediction_length: int = 8,
        window_stride: int | None = None,
        eps: float = 1e-5,
        shuffle_buffer_size: int = 100_000,
        num_epochs: int = 10,
        rank: int = 0,
        world_size: int = 1,
        base_seed: int = 42,
        split: str = "train",
    ):
        self.data_root_path = data_root_path
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_size = self.context_length + self.prediction_length
        self.window_stride = max(1, window_stride or (self.window_size // 2))
        self.eps = eps
        self.shuffle_buffer_size = shuffle_buffer_size
        self.num_epochs = num_epochs
        self.split = split
        self.rank = rank
        self.world_size = world_size
        self.base_seed = base_seed

        # Discover all CSV files once
        self.files = sorted(Path(self.data_root_path).rglob("*.csv"))
        assert len(self.files) > 0, \
            f"No CSV files found in {self.data_root_path}"

        self.file_window_counts = self._estimate_file_window_counts()

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    @staticmethod
    def _pick_val_subjects(all_subjects: list, seed: int = 0) -> set:
        """Deterministically choose held-out validation subjects.
        Rule: hold out 1 subject if dataset has <10 subjects, else 2."""
        sorted_subs = sorted(all_subjects)           # stable order
        rng = random.Random(seed)
        rng.shuffle(sorted_subs)
        n_val = 1 if len(sorted_subs) < 10 else 2
        return set(sorted_subs[:n_val])

    @staticmethod
    def _stable_int_hash(value: str, max_value: int = 0xFFFFFF) -> int:
        digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest()
        return int(digest, 16) & max_value

    def _num_windows(self, series_len: int) -> int:
        if series_len < self.window_size:
            return 0
        return 1 + (series_len - self.window_size) // self.window_stride

    def _estimate_file_window_counts(self) -> dict[str, int]:
        counts = {}
        for csv_path in self.files:
            counts[str(csv_path)] = self._estimate_windows_for_file(csv_path)
        return counts

    def _estimate_windows_for_file(self, csv_path) -> int:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return 0

        subject_col = None
        for col in df.columns:
            if col.lower() in self.META_COLS and "subject" in col.lower():
                subject_col = col
                break

        meta_cols_present = {c for c in df.columns if c.lower() in self.META_COLS}
        sensor_df = df.drop(columns=list(meta_cols_present), errors="ignore") \
                      .select_dtypes(include=[np.number])
        if sensor_df.empty:
            return 0

        if subject_col is not None and subject_col in df.columns:
            all_subjects = sorted(df[subject_col].dropna().unique().tolist())
            if len(all_subjects) == 0:
                return 0

            file_seed = self._stable_int_hash(str(csv_path))
            val_subjects = self._pick_val_subjects(all_subjects, seed=file_seed)
            train_subjects = set(all_subjects) - val_subjects
            if self.split == "train":
                mask_rows = df[subject_col].isin(train_subjects)
            else:
                mask_rows = df[subject_col].isin(val_subjects)
            sensor_df = sensor_df[mask_rows.values].reset_index(drop=True)
        else:
            sensor_df = sensor_df.reset_index(drop=True)

        if sensor_df.empty:
            return 0

        total = 0
        for col in sensor_df.columns:
            total += self._num_windows(len(sensor_df[col]))
        return total

    def _assign_files_balanced(self, files_shuffled: list, world_size: int) -> list[list[Path]]:
        file_sizes = sorted(
            files_shuffled,
            key=lambda p: self.file_window_counts.get(str(p), 0),
            reverse=True,
        )
        buckets = [[] for _ in range(world_size)]
        loads = [0 for _ in range(world_size)]

        for f in file_sizes:
            target = min(range(world_size), key=lambda i: loads[i])
            buckets[target].append(f)
            loads[target] += self.file_window_counts.get(str(f), 0)
        return buckets

    def parse_and_normalize(self, csv_path, rng: random.Random | None = None):
        """
        Parse a CSV sensor file.
        - Subject-level train/val split when a 'subject_id' column exists.
          · train split: all subjects EXCEPT the held-out ones
          · val   split: held-out subjects only
        - Metadata columns (subject_id, activity_id, …) are excluded from
          the sensor signals.
        """
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Skipping {csv_path} due to read error: {e}")
            return

        # ── Identify subject column (case-insensitive) ────────────────────
        subject_col = None
        for col in df.columns:
            if col.lower() in self.META_COLS and "subject" in col.lower():
                subject_col = col
                break

        # ── Sensor columns = numeric, not metadata ────────────────────────
        meta_cols_present = {c for c in df.columns if c.lower() in self.META_COLS}
        sensor_df = df.drop(columns=list(meta_cols_present), errors="ignore") \
                      .select_dtypes(include=[np.number])
        if sensor_df.empty:
            return

        # ── Subject-level split ───────────────────────────────────────────
        if subject_col is not None and subject_col in df.columns:
            all_subjects = sorted(df[subject_col].dropna().unique().tolist())
            if len(all_subjects) == 0:
                return

            file_seed = self._stable_int_hash(str(csv_path))
            val_subjects = self._pick_val_subjects(all_subjects, seed=file_seed)
            train_subjects = set(all_subjects) - val_subjects

            if self.split == "train":
                mask_rows = df[subject_col].isin(train_subjects)
            else:
                mask_rows = df[subject_col].isin(val_subjects)

            subject_sensor_df = sensor_df[mask_rows.values].reset_index(drop=True)
        else:

            subject_sensor_df = sensor_df.reset_index(drop=True)

        if subject_sensor_df.empty:
            return

        # ── Yield sliding windows per sensor column ───────────────────────
        if rng is None:
            rng = random.Random(0)

        columns = list(subject_sensor_df.columns)
        rng.shuffle(columns)
        for col in columns:
            vals = subject_sensor_df[col].to_numpy(dtype=np.float32)
            obs_mask = ~np.isnan(vals)
            if not obs_mask.any():
                continue

            mean = float(np.mean(vals[obs_mask]))
            std  = float(np.std(vals[obs_mask]))
            std  = max(std, self.eps)
            vals = (vals - mean) / std
            vals = np.nan_to_num(vals, nan=0.0)

            starts = list(range(0, len(vals) - self.window_size + 1, self.window_stride))
            if len(starts) == 0:
                continue
            rng.shuffle(starts)

            for i in starts:
                past      = vals[i : i + self.context_length]
                past_mask = obs_mask[i : i + self.context_length]
                sample = {
                    "past_values":        past,
                    "past_observed_mask": past_mask.astype(np.bool_),
                    "series_mean":        mean,
                    "series_std":         std,
                    "series_id":          col,
                    "file":               str(csv_path),
                }
                yield sample

    def _sample_iterator(self, files_to_process, rng: random.Random):
        for csv_path in files_to_process:
            yield from self.parse_and_normalize(csv_path, rng=rng)

    def __iter__(self):
        import torch.distributed as dist
        
        rank = self.rank
        world_size = self.world_size

        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        for epoch in range(self.num_epochs):
            files_shuffled = list(self.files)
            seed = self.base_seed + epoch
            random.Random(seed).shuffle(files_shuffled)

            file_assignments = self._assign_files_balanced(files_shuffled, world_size)
            files_for_rank = file_assignments[rank]

            rng = random.Random(seed + rank * 1000)
            rng.shuffle(files_for_rank)
            it = self._sample_iterator(files_for_rank, rng=rng)

            if self.shuffle_buffer_size <= 0:
                yield from it
            else:
                buffer = []
                for _ in range(self.shuffle_buffer_size):
                    try:
                        buffer.append(next(it))
                    except StopIteration:
                        break
                rng.shuffle(buffer)
                for sample in it:
                    idx = rng.randrange(len(buffer))
                    yield buffer[idx]
                    buffer[idx] = sample
                rng.shuffle(buffer)
                for sample in buffer:
                    yield sample


def pad_to_length(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.size(1) < length:
        pad = length - x.size(1)
        x = torch.cat([x, torch.zeros(x.size(0), pad, x.size(2), device=x.device)], dim=1)
    elif x.size(1) > length:
        x = x[:, :length, :]
    return x

class TimeSeriesCollator:
    def __init__(self, context_length: int, patch_length: int, prediction_length: int = 0):
        self.context_length = context_length
        self.patch_length = patch_length
        self.prediction_length = prediction_length

    def __call__(self, features: list) -> dict:
        past = torch.stack([
            torch.nn.functional.pad(
                torch.tensor(f['past_values'], dtype=torch.float),
                (0, self.context_length - len(f['past_values'])),
                value=0.0,
            ) for f in features
        ]).unsqueeze(-1)
        past = pad_to_length(past, self.context_length)
        mask = torch.stack([
            torch.tensor(f['past_observed_mask'], dtype=torch.bool) for f in features
        ]).unsqueeze(-1)
        mask = pad_to_length(mask, self.context_length)

        batch = {'past_values': past, 'past_observed_mask': mask}
        if self.prediction_length > 0 and 'future_values' in features[0]:
            fut = torch.stack([
                torch.tensor(f['future_values'], dtype=torch.float) for f in features
            ]).unsqueeze(-1)
            batch['future_values'] = fut
        return batch




def get_tspulse_model(args, actual_num_channels):

    cfg2 = load_config("config_clf.json")
    
    ts_cfg = TSPulseConfig(**cfg2)
    print("cfg2 fuse_fft:", cfg2.get("fuse_fft"))
    print("ts_cfg fuse_fft:", ts_cfg.fuse_fft)

    ts_cfg.context_length = args.context_length      # 512
    ts_cfg.prediction_length = 0                     # SSL / classification

    # Safety check
    assert args.context_length % args.patch_length == 0, \
        "context_length must be divisible by patch_length"

    # Patch geometry
    base_patches = args.context_length // args.patch_length
    # fuse_fft=True concatenates FFT tokens with time tokens → doubles effective patch count
    ts_cfg.patch_length = args.patch_length
    ts_cfg.patch_stride = args.patch_length
    ts_cfg.num_patches = base_patches * 2 if ts_cfg.fuse_fft else base_patches

    # Clamp mask_block_length to at most half the time-domain patches
    # (prevents block mask spanning the entire sequence for short context lengths)
    max_block = max(1, base_patches // 2)
    if ts_cfg.mask_block_length > max_block:
        print(f"[get_tspulse_model] Overriding mask_block_length: "
              f"{ts_cfg.mask_block_length} → {max_block} (base_patches={base_patches})")
        ts_cfg.mask_block_length = max_block

    # Encoder geometry
    ts_cfg.num_patches_layerwise = [
        ts_cfg.num_patches for _ in range(ts_cfg.num_layers)
    ]

    # Decoder geometry
    if ts_cfg.decoder_num_layers > 0:
        ts_cfg.decoder_num_patches_layerwise = [
            ts_cfg.num_patches for _ in range(ts_cfg.decoder_num_layers)
        ]

    # Channels
    ts_cfg.num_input_channels = actual_num_channels
    model = TSPulseForReconstruction(ts_cfg)

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"TSPulse Model: {total_params:,} total parameters, {trainable_params:,} trainable")
    return model

def prepare_model_inputs(batch):
    model_inputs = {
        "past_values": batch["past_values"],
        "past_observed_mask": batch["past_observed_mask"]
    }
    if "future_values" in batch:
        model_inputs["future_values"] = batch["future_values"]
    return model_inputs

def pretrain(args, model, dset_train, dset_val):
    lr = args.learning_rate
    logger.info(f"Starting TSPulse dual-domain pretraining with LR: {lr}")

    steps_per_epoch = max(1, int(getattr(args, "steps_per_epoch", 100)))
    logs_per_epoch = max(1, int(getattr(args, "logs_per_epoch", 1)))
    logging_steps = max(1, steps_per_epoch // logs_per_epoch)
    eval_steps = logging_steps
    save_steps = logging_steps

    logger.info(
        f"Schedule: steps/epoch={steps_per_epoch}, target_epochs={getattr(args, 'target_epochs', 'NA')}, "
        f"max_steps={args.max_steps}, logs/epoch={logs_per_epoch}, logging_steps={logging_steps}, "
        f"eval_steps={eval_steps}, save_steps={save_steps}"
    )

    ckpt_dir = os.path.join(args.save_dir, "checkpoint")
    trainer_args = TrainingArguments(
        output_dir=ckpt_dir,
        overwrite_output_dir=False,
        learning_rate=lr,
        max_steps=args.max_steps,
        warmup_steps=int(args.max_steps * 0.1),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=args.num_workers,
        seed=args.random_seed,
        logging_dir=os.path.join(args.save_dir, "logs"),
        logging_strategy="steps",
        logging_steps=logging_steps,
        eval_strategy="steps" if dset_val else "no",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=None,
        load_best_model_at_end=bool(dset_val),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        weight_decay=0.01,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        bf16=True,
        ignore_data_skip=True,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    # Early stopping callback
    callbacks = []
    if args.early_stopping and dset_val:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=10,
            early_stopping_threshold=0.001
        ))

    collator = TimeSeriesCollator(
        context_length=args.context_length,
        patch_length=args.patch_length,
        prediction_length=args.forecast_length
    )

    class TSPulseDualDomainTrainer(Trainer):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if not dist.is_initialized() or dist.get_rank() == 0:
                self.loss_log_path = os.path.join(
                    self.args.output_dir, "step_losses.txt"
                )
                with open(self.loss_log_path, "w") as f:
                    f.write("step,train_loss,eval_loss,learning_rate\n")

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            mi = {
                "past_values": inputs["past_values"],
                "past_observed_mask": inputs["past_observed_mask"],
            }

            outputs = model(**mi, return_loss=True)
            loss = outputs.loss

            if loss is None:
                loss = torch.tensor(
                    0.0,
                    device=mi["past_values"].device,
                    requires_grad=True
                )

            if not loss.dim():
                loss = loss.view(1)

            return (loss, outputs) if return_outputs else loss

        def log(self, logs: dict, start_time=None):
            # Call HF Trainer log FIRST
            super().log(logs, start_time)

            # Rank-0 only
            if dist.is_initialized() and dist.get_rank() != 0:
                return

            step = self.state.global_step

            train_loss = logs.get("loss")
            eval_loss = logs.get("eval_loss")
            lr = logs.get("learning_rate")

            msg = []
            if train_loss is not None:
                msg.append(f"train_loss={train_loss:.4f}")
            if eval_loss is not None:
                msg.append(f"val_loss={eval_loss:.4f}")
            if lr is not None:
                msg.append(f"lr={lr:.6f}")

            if msg:
                print(f"[step {step}] " + " | ".join(msg))

            # Optional CSV logging
            if not hasattr(self, "loss_log_path"):
                return

            with open(self.loss_log_path, "a") as f:
                f.write(
                    f"{step},"
                    f"{train_loss if train_loss is not None else ''},"
                    f"{eval_loss if eval_loss is not None else ''},"
                    f"{lr if lr is not None else ''}\n"
                )

    trainer = TSPulseDualDomainTrainer(
        model=model,
        args=trainer_args,
        train_dataset=dset_train,
        eval_dataset=dset_val,
        data_collator=collator,
        callbacks=callbacks,
    )

    # Resume logic
    last_ckpt = None
    if os.path.isdir(ckpt_dir) and len(os.listdir(ckpt_dir)) > 0:
        last_ckpt = get_last_checkpoint(ckpt_dir)
    if last_ckpt:
        logger.info(f"Resuming training from checkpoint: {last_ckpt}")
        state_file = os.path.join(last_ckpt, "trainer_state.json")
        if os.path.isfile(state_file):
            logger.info("  → stripping unsupported fields from trainer_state.json")
            with open(state_file, "r") as f:
                state = json.load(f)
            # remove keys introduced after Transformers 4.40
            for key in ("best_global_step",): # Only remove keys that you are certain cause issues.
                if key in state:
                    state.pop(key)
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)

    logger.info("Starting TSPulse dual-domain pretraining...")
    train_result = trainer.train(resume_from_checkpoint=last_ckpt)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    logger.info(f"Training completed. Final metrics: {metrics}")

    if dset_val:
        eval_metrics = trainer.evaluate(dset_val)
        trainer.log_metrics("eval", eval_metrics)
        logger.info(f"Final validation metrics: {eval_metrics}")

    save_path = os.path.join(args.save_dir, "tspulse_dual_domain_pretrained")
    trainer.save_model(save_path)
    logger.info(f"TSPulse model saved to: {save_path}")
    return save_path




def inference_and_evaluation(args, model_path, dset_test):
    model = get_model(model_path=model_path)
    
    temp_dir = tempfile.mkdtemp()
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=args.batch_size,
            seed=args.random_seed,
            report_to="none",
        ),
    )
    
    logger.info("="*20 + " TSPulse Test Results " + "="*20)
    output = trainer.evaluate(dset_test)
    logger.info(f"Test metrics: {output}")
    
    predictions_dict = trainer.predict(dset_test)
    predictions_np = predictions_dict.predictions[0]
    
    if len(predictions_dict.predictions) > 1:
        backbone_embedding = predictions_dict.predictions[1]
        logger.info(f"Predictions shape: {predictions_np.shape}")
        logger.info(f"Backbone embeddings shape: {backbone_embedding.shape}")
    
    plot_path = os.path.join(args.save_dir, "plots")
    plot_predictions(
        model=trainer.model,
        dset=dset_test,
        plot_dir=plot_path,
        plot_prefix="tspulse_test_inference",
        channel=0,
    )
    logger.info(f"Plots saved to: {plot_path}")
    
    return output

def count_total_windows_csv(
    data_root_path: str,
    context_length: int,
    prediction_length: int = 0,
):
    total = 0
    for csv_path in Path(data_root_path).rglob("*.csv"):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        # Match dataset logic
        df = df.select_dtypes(include=[np.number])
        if df.empty:
            continue

        window = context_length + prediction_length

        for col in df.columns:
            vals = df[col].dropna()
            L = len(vals)
            total += max(0, L - window + 1)

    return total


import json
import yaml
def load_config(path):
    if path.endswith(".yaml") or path.endswith(".yml"):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    elif path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    else:
        raise ValueError("Config file must be .yaml or .json")


def main():
    #  Initialize distributed
    dist.init_process_group(backend="nccl")

    #  Bind THIS process to its GPU
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    #  Debug prints (now correct)
    print("Global Rank:", dist.get_rank())
    print("Local Rank:", local_rank)
    print("Using device:", device)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))
    print("torch.cuda.device_count():", torch.cuda.device_count())
    print(
        "torch.cuda.get_device_name():",
        torch.cuda.get_device_name(local_rank)
    )

    #  Logging
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    

    args = get_ttm_args()
    # cng = load_config("config.json")
    
    # WORKING TSPulse configuration
    args.context_length = 512
    args.forecast_length = 0
    args.patch_length = 16
    args.overlap_ratio = 0.75
    window_size = args.context_length + args.forecast_length
    args.window_stride = max(1, int(round(window_size * (1.0 - args.overlap_ratio))))
    args.batch_size = 4096
    args.early_stopping = False   # default from get_ttm_args() is 1 (True) — must explicitly disable
    args.num_epochs = 100000
    args.target_epochs = 40
    args.logs_per_epoch = 1
    logger.info("="*50)
    logger.info("TSPulse Dual-Domain Pretraining Configuration")
    logger.info("="*50)
    logger.info(f"Context Length: {args.context_length}")
    logger.info(f"Forecast Length: {args.forecast_length}")
    logger.info(f"Patch Length: {args.patch_length}")
    logger.info(f"Patches: {args.context_length // args.patch_length}")
    logger.info(f"Overlap Ratio: {args.overlap_ratio:.2f}")
    logger.info(f"Window Stride: {args.window_stride}")
    logger.info(f"Model Dimension: {args.d_model}")
    logger.info(f"Encoder Layers: {args.num_layers}")
    logger.info(f"Decoder Layers: {args.decoder_num_layers}")
    logger.info(f"Dual-Domain: Enabled (FFT addition)")
    logger.info("="*50)
    
    set_seed(args.random_seed)

    # OPTION 1: Use with your energy dataset
    args.data_root_path = "/path/to/pretrain_dataset"
    args.save_dir = "./tspulse_pretrained"
    os.makedirs(args.save_dir, exist_ok=True)

    if hasattr(args, 'data_root_path') and args.data_root_path:
        logger.info(f"Loading energy datasets from: {args.data_root_path}")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        dset_train = StreamingTimeSeriesDataset(
            data_root_path=args.data_root_path,
            context_length=args.context_length,
            prediction_length=0,
            window_stride=args.window_stride,
            split="train",
            shuffle_buffer_size=240_000,   # ctx=128 is 4× smaller than ctx=512; safe to use 240K (vs 60K for ctx=512)
            num_epochs=args.num_epochs,
            rank=rank,
            world_size=world_size,
        )

        dset_val = StreamingTimeSeriesDataset(
            data_root_path=args.data_root_path,
            context_length=args.context_length,
            prediction_length=0,
            window_stride=args.window_stride,
            split="val",
            shuffle_buffer_size=0,   # no shuffle needed for val
            num_epochs=1,
            rank=rank,
            world_size=world_size,
        )
        dset_test = None

        # Use actual number of channels (1 for univariate energy data)
        actual_num_channels = 1

        if rank == 0:
            total_windows_local = int(sum(dset_train.file_window_counts.values()))
        else:
            total_windows_local = None

        obj_list = [total_windows_local]
        dist.broadcast_object_list(obj_list, src=0)
        total_windows = int(obj_list[0])

        args.number_of_gpus = world_size
        effective_batch = args.batch_size * args.number_of_gpus
        args.steps_per_epoch = max(1, math.ceil(total_windows / effective_batch))
        args.max_steps = args.steps_per_epoch * args.target_epochs

        logger.info(
            f"Stride-aware train windows: {total_windows:,} "
            f"(stride={args.window_stride}, overlap={args.overlap_ratio:.2f})"
        )
        logger.info(
            f"World size: {args.number_of_gpus} | Effective batch: {effective_batch} | "
            f"Steps/epoch: {args.steps_per_epoch} | Epochs: {args.target_epochs} | Max steps: {args.max_steps}"
        )


    # Get TSPulse model with correct channel configuratio
    model = get_tspulse_model(args, actual_num_channels)

    # Pretrain the model
    model_save_path = pretrain(args, model, dset_train, dset_val)
    logger.info("="*50)
    logger.info("TSPulse Dual-Domain Pretraining Completed!")
    logger.info(f"Model saved to: {model_save_path}")
    logger.info("="*50)

    # Run inference and evaluation
    if dset_test is not None and len(dset_test) > 0:
        inference_and_evaluation(args, model_save_path, dset_test)
        logger.info("Inference and evaluation completed!")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()