# HART: A Pretrained Time-Series Foundation Model and Benchmark for Human Activity Recognition

This repository contains the code accompanying our paper. It is organised into four stages: **data preprocessing**, **HART pre-training**, **HART fine-tuning (LOSO evaluation)**, and **baseline benchmarking**.

```
anonymous_github/
├── environment.yml
├── data_preprocessing/
│   ├── whar_preprocessor.py # WHAR dataset preprocessing pipeline
│   └── Resampling.py        # Polyphase FIR resampling of all HAR datasets to 30 Hz
├── pretraining_HART/
│   ├── config_clf.json      # HART model architecture config
│   └── pretrain.py          # Distributed SSL pre-training script
├── finetuning_HART/
│   ├── pretrained_checkpoint/ # Provided best pre-trained HART model
│   │   ├── config.json
│   │   └── model.safetensors
│   ├── finetune.py          # LOSO fine-tuning & evaluation script for HART
│   ├── HAR_dataloader.py    # HAR dataset windowing & loading utilities
│   └── run_all.sh           # Convenience script — runs all datasets in parallel on 2 GPUs
└── benchmarking/            # Baseline models evaluation 
    ├── HAR_models/          # Deep learning HAR baselines (HARNet, TinyHAR)
    ├── TSFMs/               # Time Series Foundational Models (MOMENT, UniTS, TSPulse)
    └── ml_models/           # Scikit-learn based machine learning models
```

---

## 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate <env_name>
```

---

## 2. Data Preprocessing

All HAR datasets used in this work are sourced from the [WHAR-datasets repository](https://github.com/teco-kit/whar-datasets). The datasets are first downloaded and preprocessed as per the WHAR pipeline. Then, they are resampled at the session level to a unified **30 Hz** before pre-training or fine-tuning.

### Step 1 — WHAR Preprocessing

```bash
python data_preprocessing/whar_preprocessor.py
```

This runs the full `whar_datasets` preprocessing pipeline (segmentation, normalisation, LOSO splits) over all supported datasets. Edit `whar_preprocessor.py` to restrict processing to specific datasets via `WHARDatasetID`. Refer to WHAR Repo for all datasets and preprocessing: [WHAR-datasets repository](https://github.com/teco-kit/whar-datasets)

### Step 2 — Resample to 30 Hz

```bash
python data_preprocessing/Resampling.py \
    --input_dir  /path/to/original_datasets \
    --output_dir /path/to/output_30hz \
    --target_fs  30
```

| Argument | Description |
|---|---|
| `--input_dir` | Directory containing the original raw CSV datasets |
| `--output_dir` | Where to save the resampled CSVs |
| `--target_fs` | Target sampling frequency (default: `30`) |


---

## 3. HART Pre-training

Pre-training uses **multi-GPU distributed training** via `torchrun`.

Before running, open `pretraining_HART/pretrain.py` and set the following paths:

```python
args.data_root_path = "/path/to/your/pretrain_dataset"   # folder of 30 Hz CSVs
args.save_dir       = "./tspulse_pretrained"              # where checkpoints are saved
```

Then launch with:

```bash
torchrun --nproc_per_node=<NUM_GPUS> pretraining_HART/pretrain.py
```

**Example (2 GPUs):**

```bash
torchrun --nproc_per_node=2 pretraining_HART/pretrain.py
```

The model architecture is controlled by `pretraining_HART/config_clf.json`. Key training hyperparameters (context length, patch size, batch size, epochs) are set directly in `main()` inside `pretrain.py`.

---

## 4. HART Fine-tuning (LOSO Evaluation)

Fine-tuning evaluates the pre-trained model on each HAR dataset using **Leave-One-Subject-Out (LOSO)** cross-validation.

> **Note on Checkpoints:** To easily reproduce the results from our paper, we provide our best pre-trained checkpoint in `finetuning_HART/pretrained_checkpoint`. By default, the scripts below use this checkpoint so you can skip the computationally expensive pre-training stage. However, if you run the pre-training stage yourself (Stage 3), you can simply replace this path with the directory of your newly generated checkpoint.

### Run a single dataset

```bash
python finetuning_HART/finetune.py \
    --dataset_name    capture24 \
    --checkpoint_path finetuning_HART/pretrained_checkpoint \
    --device          cuda:0 \
    --context_length  120 \
    --hop_length      30 \
    --epochs          100 \
    --patience        15 \
    --test_set_dir    /path/to/test_set \
    --output_dir      results
```

| Argument | Description |
|---|---|
| `--dataset_name` | Name of the HAR dataset to evaluate |
| `--checkpoint_path` | Path to the pre-trained HART checkpoint directory |
| `--device` | `cpu`, `cuda`, or `cuda:0` |
| `--context_length` | Window length fed to the model (default: `512`) |
| `--hop_length` | Stride between windows — `30` gives 75% overlap with `context_length=120` |
| `--epochs` | Max fine-tuning epochs (default: `100`) |
| `--patience` | Early stopping patience (default: `15`) |
| `--test_set_dir` | Folder containing per-dataset HAR CSV files |
| `--output_dir` | Directory where result CSVs are written |

### Run all datasets in parallel (recommended)

`run_all.sh` manages a two-GPU job queue — both GPUs stay busy at all times and a free GPU immediately picks up the next dataset.

```bash
bash finetuning_HART/run_all.sh \
    finetuning_HART/pretrained_checkpoint \
    cuda:0 \
    100 \
    30 \
    <conda_env_name> \
    /path/to/test_set \
    120
```

**Positional arguments (all optional — defaults shown in the script):**

| Position | Argument | Default |
|---|---|---|
| 1 | Checkpoint path | `pretrained_checkpoint` |
| 2 | Device (kept for compatibility) | `cuda:0` |
| 3 | Epochs | `100` |
| 4 | Hop length (window stride) | `30` |
| 5 | Conda environment name | `tspulse_env` |
| 6 | Test set directory | `test_set/` next to script |
| 7 | Context length | `120` |

Results are written to `results/<checkpoint_name>_ep<N>_ctx<L>/` and per-dataset logs to `results/logs_*/`. To monitor a running job live:

```bash
tail -f results/logs_<checkpoint>_ep100_120/<dataset>.log
```

---

## 5. Baseline Benchmarking

The baseline models used for benchmarking against our approach are included in the `benchmarking/` folder (formerly named `Final_HAR_Code/`). The benchmarking setup applies the exact same LOSO validation and evaluation framework as used for HART.

The setup is split into three main categories:

### Machine Learning Baselines (`benchmarking/ml_models/`)
Contains traditional time series classification models via `sktime` and `scikit-learn`, including:
- **Dummy Classifier** (uniform baseline)
- **Rocket / MiniRocket** algorithms
- **Arsenal**
- **KNN Time Series Classifier** (with DTW distance)

- Run `all_sktime.sh` to execute the evaluation across all HAR datasets (uses `ml_sktime_person.py` or `ml_sktime.py`).
- Uses `har_dataset_loader.py` for uniform data loading.

### Deep Learning HAR Baselines (`benchmarking/HAR_models/`)
Contains established deep learning baselines tailored for HAR:
- **HARNet**: Run via `all_harnet.sh` (which invokes `harnet_har.py`).
- **TinyHAR**: Included as an interactive Jupyter Notebook (`TinyHAR.ipynb`).

### Time Series Foundation Models (`benchmarking/TSFMs/`)
Contains generic state-of-the-art foundation models for time series adapted for the HAR task:
- **MOMENT**: Run via `all_moment.sh`.
- **UniTS**: Run via `all_units.sh`.
- **TSPulse**: Baseline model implementation.

Each folder includes a `har_dataset_loader.py` tool adapted for that specific baseline to ensure inputs conform to the expected format.
