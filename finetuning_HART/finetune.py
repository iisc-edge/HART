import os
import sys
_dev = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--device"), "cuda")
_gpu_idx = _dev.split(":")[-1] if ":" in _dev else "0"
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_idx

os.environ["TRANSFORMERS_OFFLINE"] = "1"   # prevent HF hub validation on local paths
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"  # reduce fragmentation OOM/segfaults
import math
import argparse
import tempfile
import warnings
import numpy as np
import pandas as pd
import torch

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split

from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed
)
from transformers.data.data_collator import default_data_collator
from transformers.trainer_utils import RemoveColumnsCollator

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataclasses import dataclass
from typing import Optional, Tuple

from tsfm_public.models.tspulse import TSPulseForClassification
from tsfm_public.models.tspulse.modeling_tspulse import (
    TSPulseLayer,
    TSPulseModelOutput,
    register_token_config_update,
)
from tsfm_public.toolkit.dataset import ClassificationDFDataset
from tsfm_public.toolkit.lr_finder import optimal_lr_finder
from tsfm_public.toolkit.time_series_classification_preprocessor import (
    TimeSeriesClassificationPreprocessor
)

from HAR_dataloader import (
    load_dataset,
    get_input_columns,
    get_label_column,
    get_subject_column,
    DATASET_CONFIGS,
)

warnings.filterwarnings("ignore")
set_seed(42)


# ============================================================
# Shared output container — HuggingFace Trainer compatible.
# Both loss and logits are always populated so Trainer can
# correctly log eval_loss on every evaluation step.
# ============================================================


@dataclass
class TSPulseOutput:
    loss:               Optional[torch.FloatTensor] = None
    logits:             Optional[torch.FloatTensor] = None
    prediction_outputs: Optional[torch.FloatTensor] = None

    def __iter__(self):
        # Trainer sees: outputs[0]=loss, outputs[1:]=predictions
        yield self.loss
        yield self.logits

    def __getitem__(self, idx):
        return (self.loss, self.logits)[idx]


# ============================================================
# Standard Head — TSPulseForClassification with
# built-in gated-attention pooling.
#
# Pipeline:
#   Backbone → gated-attention pool → Linear → CE loss
#
# Freeze policy:
#   Frozen  : entire backbone
#   Trainable: time_encoding + fft_encoding + classification head
#
# Loss is computed explicitly here (not delegated to inner model)
# so Trainer always receives loss → eval_loss logged.
# ============================================================

class TSPulseForHAR(torch.nn.Module):
    """ TSPulse classifier for HAR — HuggingFace Trainer compatible."""

    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int,
        num_input_channels: int,
        config_overrides: dict,
    ):
        super().__init__()

        # Load full model with built-in gated-attention head
        overrides = dict(config_overrides)
        overrides["loss"] = "cross_entropy"
        self._model = TSPulseForClassification.from_pretrained(
            checkpoint_path, **overrides
        )
        cfg = self._model.config

        print(f"\n[HART] head_aggregation      = {cfg.head_aggregation_dim}")
        print(f"[HART] gated_attention_act   = {cfg.head_gated_attention_activation}")
        print(f"[HART] num_input_channels    = {num_input_channels}")
        print(f"[HART] num_classes           = {num_classes}")

        self._loss_fn = torch.nn.CrossEntropyLoss()

        # ── Freeze the full backbone ───────────────────────────────────────
        for p in self._model.backbone.parameters():
            p.requires_grad = False

        # ── Unfreeze both encoding layers ────────────
        for p in self._model.backbone.time_encoding.parameters():
            p.requires_grad = True
        for p in self._model.backbone.fft_encoding.parameters():
            p.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"[HART] Trainable: {trainable:,} / {total:,}\n")

    def forward(
        self,
        past_values:        torch.Tensor,
        target_values:      Optional[torch.Tensor] = None,
        past_observed_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> TSPulseOutput:
        """
        Args:
            past_values       : [B, S, C]
            target_values     : [B]  class indices (LongTensor)
            past_observed_mask: [B, S, C] optional
        Returns:
            TSPulseOutput with .loss, .logits, .prediction_outputs
        """
        
        out = self._model(
            past_values=past_values,
            target_values=None,                    
            past_observed_mask=past_observed_mask,
        )

        logits = (
            out.prediction_outputs
            if hasattr(out, "prediction_outputs") and out.prediction_outputs is not None
            else out[1]
        )

        loss = None
        if target_values is not None:
            loss = self._loss_fn(logits, target_values.long())

        return TSPulseOutput(loss=loss, logits=logits, prediction_outputs=logits)


# ============================================================
# Argument Parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="HART HAR LOSO Fine-tuning"
    )

    parser.add_argument(
        "--dataset_name", type=str, required=True,
        help=f"Dataset name. One of: {sorted(DATASET_CONFIGS.keys())}"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to pretrained TSPulse checkpoint"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device: cpu | cuda | cuda:0"
    )
    parser.add_argument("--epochs",         type=int, default=100)
    parser.add_argument("--batch_size",     type=int, default=3096)
    parser.add_argument("--patience",       type=int, default=15,
                        help="EarlyStoppingCallback patience (default: 15)")
    parser.add_argument(
        "--hop_length", type=int, default=128,
        help="Window hop (stride) for overlapping windowing. "
             "128 = 75%% overlap with context_length=512. "
             "Use context_length value to disable overlap."
    )
    parser.add_argument(
        "--context_length", type=int, default=512,
        help="Sequence / window length fed to the model."
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Directory to store output CSVs"
    )
    parser.add_argument(
        "--test_set_dir", type=str, default=None,
        help="Path to the folder containing HAR CSV files. "
             "Defaults to the 'test_set/' subfolder next to HAR_dataloader.py."
    )

    return parser.parse_args()


# ============================================================
# Metrics
# ============================================================

def classification_metrics(y_true, y_pred, average="macro"):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred,
                                     average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred,
                               average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred,
                             average=average, zero_division=0)
    }


# ============================================================
# Dataset Preparation
# ============================================================

def prepare_dataset(
    df_base, df_test,
    input_columns, label_column, all_classes,
    context_length: int = 512,
):
    """Fit scaler on train, apply to both train and test, return HF datasets."""

    tsp = TimeSeriesClassificationPreprocessor(
        input_columns=input_columns,
        label_column=label_column,
        scaling=True,
    )

    tsp.train(df_base)
    tsp.label_encoder.classes_ = np.array(all_classes)

    df_train_prep = tsp.preprocess(df_base)
    df_test_prep  = tsp.preprocess(df_test)

    _ds_kwargs = dict(
        id_columns=[],
        timestamp_column=None,
        input_columns=input_columns,
        label_column=label_column,
        context_length=context_length,
        static_categorical_columns=[],
        stride=1,
        enable_padding=False,
        full_series=True,
    )

    base_dataset = ClassificationDFDataset(df_train_prep, **_ds_kwargs)
    test_dataset = ClassificationDFDataset(df_test_prep,  **_ds_kwargs)

    # 90/10 train/val split
    dataset_size = len(base_dataset)
    val_size   = max(1, int(0.1 * dataset_size))
    train_size = dataset_size - val_size
    train_dataset, valid_dataset = random_split(base_dataset, [train_size, val_size])

    print(f"  Fold sizes — train: {train_size}, val: {val_size}, test: {len(test_dataset)}")
    return train_dataset, valid_dataset, test_dataset, tsp

# ============================================================
# Model Factory
# ============================================================

def load_model(checkpoint_path, tsp, num_targets, context_length, device):
    """Instantiate TSPulseForHAR with classification config overrides."""

    config_overrides = {
        # ── Classification hyperparameters ──────────────────────────────
        "head_gated_attention_activation": "softmax",
        "channel_virtual_expand_scale":    1,     
        "mask_ratio":                       0.3,
        "head_reduce_d_model":              1,
        "disable_mask_in_classification_eval": True,
        "fft_time_consistent_masking":      True,
        "decoder_mode":                     "common_channel",
        "head_aggregation_dim":             "patch",
        "head_aggregation":                 None,
        # ── Task-specific ───────────────────────────────────────────────
        "num_input_channels":  tsp.num_input_channels,
        "num_targets":         num_targets,
        "loss":                "cross_entropy",
        "ignore_mismatched_sizes": True,
    }

    model = TSPulseForHAR(
        checkpoint_path=checkpoint_path,
        num_classes=num_targets,
        num_input_channels=tsp.num_input_channels,
        config_overrides=config_overrides,
    ).to(device)

    return model


# ============================================================
# Training + Evaluation
# ============================================================

def train_and_evaluate(df_train, df_test, args, input_columns, label_column, all_classes):

    train_ds, val_ds, test_ds, tsp = prepare_dataset(
        df_train, df_test, input_columns, label_column, all_classes,
        context_length=args.context_length,
    )

    # Auto-scale batch size for high-channel datasets to avoid CUDA OOM.
    n_ch = tsp.num_input_channels
    if n_ch <= 9:
        effective_batch = args.batch_size
    elif n_ch <= 18:
        effective_batch = min(args.batch_size, 1024)
    elif n_ch <= 36:
        effective_batch = min(args.batch_size, 512)
    elif n_ch <= 72:
        effective_batch = min(args.batch_size, 256)
    else:
        effective_batch = min(args.batch_size, 64)

    if effective_batch != args.batch_size:
        print(f"[OOM-guard] {n_ch} input channels → reducing batch {args.batch_size} → {effective_batch}")

    model = load_model(
        args.checkpoint_path,
        tsp,
        len(all_classes),          
        args.context_length,
        args.device,
    )

    lr, model = optimal_lr_finder(
        model,
        train_ds,
        batch_size=effective_batch,
    )

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = OneCycleLR(
        optimizer, lr,
        epochs=args.epochs,
        steps_per_epoch=max(1, math.ceil(len(train_ds) / effective_batch)),
    )

    training_args = TrainingArguments(
        output_dir=tempfile.mkdtemp(),
        learning_rate=lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=effective_batch,
        per_device_eval_batch_size=effective_batch,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        label_names=["target_values"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        optimizers=(optimizer, scheduler),
    )

    trainer.train()

    # Test Evaluation
    # Trainer stores logits directly in .predictions (via outputs.logits attribute)
    raw_preds = trainer.predict(test_ds).predictions
    # .predictions may be a tuple (logits,) or a plain ndarray [N, C]
    preds = raw_preds[0] if isinstance(raw_preds, tuple) else raw_preds
    y_pred = np.argmax(preds, axis=1)

    dataloader = DataLoader(
        test_ds,
        batch_size=effective_batch,
        shuffle=False,
        collate_fn=RemoveColumnsCollator(
            data_collator=default_data_collator,
            signature_columns=["target_values"],
            logger=None,
            description=None,
            model_name="temp",
        )
    )

    y_true = np.concatenate(
        [batch["target_values"].numpy() for batch in dataloader]
    )

    result = classification_metrics(y_true, y_pred)

    
    del trainer, model, optimizer, scheduler
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return result


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    dataset_name = args.dataset_name
    checkpoint_name = os.path.basename(os.path.normpath(args.checkpoint_path))
    run_output_dir = os.path.join(args.output_dir, f"{checkpoint_name}_ep{args.epochs}_ctx{args.context_length}")
    os.makedirs(run_output_dir, exist_ok=True)
    output_csv_path = os.path.join(run_output_dir, f"{dataset_name}.csv")


    if args.device.startswith("cuda"):
        args.device = "cuda:0"
    device = torch.device(args.device)
    print(f"Using device: {args.device}  (CUDA_VISIBLE_DEVICES={_gpu_idx})")

    # ── Load + window the dataset via HAR_dataloader ──────────────────
    input_columns = get_input_columns(dataset_name)
    label_column  = get_label_column(dataset_name)
    subject_col   = get_subject_column(dataset_name)

    # hop_length controls overlap: 128 → 75% overlap with context_length=512
    # window_length must match context_length (512) for the pretrained model
    load_kwargs = dict(
        window_length=args.context_length,
        hop_length=args.hop_length,
    )
    if args.test_set_dir:
        load_kwargs["test_set_dir"] = args.test_set_dir

    df_cleaned = load_dataset(dataset_name, **load_kwargs)

    print(f"Loaded '{dataset_name}': {len(df_cleaned)} windows, "
          f"{df_cleaned[subject_col].nunique()} subjects, "
          f"{df_cleaned[label_column].nunique()} classes")
    print(f"Window length={args.context_length}, hop={args.hop_length}")

    # ── Keep only activities common to ALL subjects (fair LOSO evaluation) ─
    per_subject_labels = df_cleaned.groupby(subject_col)[label_column].apply(set)
    common_labels = set.intersection(*per_subject_labels)
    before = len(df_cleaned)
    df_cleaned = df_cleaned[df_cleaned[label_column].isin(common_labels)].reset_index(drop=True)
    print(f"Common activities across all subjects ({len(common_labels)}): {sorted(common_labels)}")
    print(f"Dropped {before - len(df_cleaned)} windows ({before} → {len(df_cleaned)})")

    # ── LOSO loop ─────────────────────────────────────────────────────────
    # Global class list — fixed across all folds so label indices are consistent
    all_classes = sorted(df_cleaned[label_column].unique())
    print(f"Global classes ({len(all_classes)}): {all_classes}")

    all_metrics = []

    for fold_idx, sid in enumerate(df_cleaned[subject_col].unique()):
        df_train = df_cleaned[df_cleaned[subject_col] != sid].reset_index(drop=True)
        df_test  = df_cleaned[df_cleaned[subject_col] == sid].reset_index(drop=True)

        print(f"\n{'='*60}")
        print(f"Fold {fold_idx+1}  |  Test subject: {sid}")
        print(f"  Train windows: {len(df_train)}  |  Test windows: {len(df_test)}")
        print(f"{'='*60}")

        metrics = train_and_evaluate(
            df_train, df_test, args,
            input_columns, label_column, all_classes,
        )
        metrics["subject_id"] = sid
        all_metrics.append(pd.DataFrame([metrics]))

        # Free GPU memory between folds to prevent OOM on high-channel datasets
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        print(f"  → Acc={metrics['accuracy']:.4f}  "
              f"F1={metrics['f1_score']:.4f}  "
              f"Prec={metrics['precision']:.4f}  "
              f"Rec={metrics['recall']:.4f}")

    results = pd.concat(all_metrics).reset_index(drop=True)
    results.to_csv(output_csv_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Dataset : {dataset_name}")
    print(f"  Subjects: {len(all_metrics)}")
    print(f"  Accuracy: {results['accuracy'].mean():.4f} ± {results['accuracy'].std():.4f}")
    print(f"  F1 Score: {results['f1_score'].mean():.4f} ± {results['f1_score'].std():.4f}")
    print(f"  Results → {output_csv_path}")
    print(f"{'='*60}")
    

if __name__ == "__main__":
    main()