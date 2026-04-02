#!/bin/bash
# ============================================================
# run_all.sh  —  Run TSPulse LOSO fine-tuning benchmark on ALL datasets
#                
# Usage:
#   bash run_all.sh [checkpoint_path] [device] [epochs] [hop_length] [conda_env] [test_set_dir] [context_length]
#
# Examples:
#   bash run_all.sh
#   bash run_all.sh /path/to/checkpoint cuda:0 200 128 tspulse_env "" 512
#   bash run_all.sh finetuning_HART/pretrained_checkpoint cuda:0 100 30 tspulse_env "" 120
#   (Or pass the path to your own newly pre-trained checkpoint below)
#
# To watch a running dataset live:
#   tail -f results/logs_<checkpoint>_ep<N>_<ctx>/<dataset>.log
# ============================================================

CHECKPOINT=${1:-"finetuning_HART/pretrained_checkpoint"}
DEVICE=${2:-"cuda:0"}         
EPOCHS=${3:-100}
HOP_LENGTH=${4:-30}           # 30 = 75% overlap with context_length=120
CONDA_ENV=${5:-"tspulse_env"}
# Path to the folder containing all dataset CSVs.
TEST_SET_DIR=${6:-"$(cd "$(dirname "${BASH_SOURCE[0]}")"/test_set && pwd)"}
CONTEXT_LENGTH=${7:-120}      # 7th arg: window/context length
PATIENCE=15
OUTPUT_DIR="results"
CHECKPOINT_NAME=$(basename "$CHECKPOINT")
# Subdirectory that finetune.py creates inside OUTPUT_DIR:
RESULT_SUBDIR="${CHECKPOINT_NAME}_ep${EPOCHS}_ctx${CONTEXT_LENGTH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/$OUTPUT_DIR/logs_${CHECKPOINT_NAME}_ep${EPOCHS}_${CONTEXT_LENGTH}"

mkdir -p "$SCRIPT_DIR/$OUTPUT_DIR/$RESULT_SUBDIR"
mkdir -p "$LOG_DIR"

# All datasets with CSVs present in test_set/
# (adl is excluded — its CSV is in a separate left_out folder)
DATASETS=(
    capture24
    daphnet
    dsads
    harsense
    harth
    hhar
    hugadb
    ku_har
    mhealth
    motion_sense
    opportunity
    realworld
    uci_har
    usc_had
    w_har
    wisdm
    wisdm_19_phone
    wisdm_19_watch
)

TOTAL=${#DATASETS[@]}

# ── Job queue state ───────────────────────────────────────────────────────────
declare -A PID_TO_GPU        # pid → gpu index (0 or 1)
declare -A PID_TO_NAME       # pid → dataset name
declare -A PID_TO_START      # pid → unix start time
declare -A DATASET_ELAPSED   # dataset name → elapsed seconds (set on completion)
FREE_GPUS=(0 1)              # both GPU slots start free
RUNNING_PIDS=()              # pids currently running in background
DONE_COUNT=0
PASS=0
FAIL=0
FAILED_LIST=()

START_TIME=$(date +%s)

echo "=============================="
echo "  TSPulse HAR — Full Benchmark"
echo "=============================="
echo "  Checkpoint  : $CHECKPOINT"
echo "  Epochs      : $EPOCHS"
echo "  Context len : $CONTEXT_LENGTH"
echo "  Hop length  : $HOP_LENGTH"
echo "  Patience    : $PATIENCE"
echo "  Conda env   : $CONDA_ENV"
echo "  Test set dir: $TEST_SET_DIR"
echo "  Results dir : $SCRIPT_DIR/$OUTPUT_DIR/$RESULT_SUBDIR/"
echo "  Log dir     : $LOG_DIR/"
echo "  Datasets    : $TOTAL"
echo "  GPUs        : 0 and 1  (job-queue, both always busy)"
echo "=============================="
echo ""

# ── harvest: record completion of a finished pid ──────────────────────────────
harvest() {
    local pid=$1 exit_code=$2
    local name="${PID_TO_NAME[$pid]}"
    local gpu="${PID_TO_GPU[$pid]}"
    local elapsed=$(( $(date +%s) - PID_TO_START[$pid] ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))

    DONE_COUNT=$(( DONE_COUNT + 1 ))
    DATASET_ELAPSED[$name]=$elapsed

    # Return GPU slot to free pool
    FREE_GPUS+=("$gpu")

    # Remove pid from RUNNING_PIDS
    local keep=()
    for p in "${RUNNING_PIDS[@]}"; do [[ $p != "$pid" ]] && keep+=("$p"); done
    RUNNING_PIDS=("${keep[@]}")

    if [[ $exit_code -eq 0 ]]; then
        echo "  ✓ DONE  [$DONE_COUNT/$TOTAL]  $name  [GPU $gpu]  time: ${mins}m ${secs}s"
        PASS=$(( PASS + 1 ))
    else
        echo "  ✗ FAIL  [$DONE_COUNT/$TOTAL]  $name  [GPU $gpu]  time: ${mins}m ${secs}s  (exit $exit_code)"
        FAIL=$(( FAIL + 1 ))
        FAILED_LIST+=("$name")
    fi
}

wait_one() {
    wait -n 2>/dev/null   
    
    for pid in "${RUNNING_PIDS[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            local name="${PID_TO_NAME[$pid]}"
            local exit_file="$LOG_DIR/.exit_${name}"
            local exit_code=0
            # Give the subshell a brief moment to flush the exit file
            for _ in 1 2 3 4 5; do
                [[ -f "$exit_file" ]] && break
                sleep 0.2
            done
            if [[ -f "$exit_file" ]]; then
                exit_code=$(< "$exit_file")
                rm -f "$exit_file"
            fi
            harvest "$pid" "$exit_code"
            return
        fi
    done
}

# ── launch: start one dataset job on a given GPU in the background ────────────
launch() {
    local dataset=$1 gpu=$2
    local log_file="$LOG_DIR/${dataset}.log"
    local exit_file="$LOG_DIR/.exit_${dataset}"
    rm -f "$exit_file"

    # Subshell: run the job then write its exit code to a file
    (
        conda run --no-capture-output -n "$CONDA_ENV" \
            python -u "$SCRIPT_DIR/finetune.py" \
                --dataset_name    "$dataset" \
                --checkpoint_path "$CHECKPOINT" \
                --device          "cuda:$gpu" \
                --epochs          "$EPOCHS" \
                --hop_length      "$HOP_LENGTH" \
                --context_length  "$CONTEXT_LENGTH" \
                --patience        "$PATIENCE" \
                --test_set_dir    "$TEST_SET_DIR" \
                --output_dir      "$SCRIPT_DIR/$OUTPUT_DIR"
        echo $? > "$exit_file"
    ) > "$log_file" 2>&1 &

    local pid=$!
    PID_TO_GPU[$pid]=$gpu
    PID_TO_NAME[$pid]=$dataset
    PID_TO_START[$pid]=$(date +%s)
    RUNNING_PIDS+=("$pid")

    # Remove this GPU from the free list
    local new_free=()
    for g in "${FREE_GPUS[@]}"; do [[ $g != "$gpu" ]] && new_free+=("$g"); done
    FREE_GPUS=("${new_free[@]}")

    echo "  → LAUNCH  $dataset  [GPU $gpu]  (tail -f $(basename "$LOG_DIR")/${dataset}.log)"
}

# ── Main job-queue loop ───────────────────────────────────────────────────────
echo "Starting job queue (2 GPU slots)..."
echo ""

IDX=0
for DATASET in "${DATASETS[@]}"; do
    IDX=$(( IDX + 1 ))
    RESULT_FILE="$SCRIPT_DIR/$OUTPUT_DIR/$RESULT_SUBDIR/${DATASET}.csv"

    # Skip if result already exists (allows resuming interrupted runs)
    if [[ -f "$RESULT_FILE" ]]; then
        echo "  [SKIP] [$IDX/$TOTAL]  $DATASET — result already exists"
        PASS=$(( PASS + 1 ))
        DONE_COUNT=$(( DONE_COUNT + 1 ))
        DATASET_ELAPSED[$DATASET]=0
        continue
    fi

    # Wait for a free GPU slot if both are busy
    while [[ ${#FREE_GPUS[@]} -eq 0 ]]; do
        wait_one
    done

    launch "$DATASET" "${FREE_GPUS[0]}"
done

# Drain any jobs still running after the queue is exhausted
echo ""
if [[ ${#RUNNING_PIDS[@]} -gt 0 ]]; then
    echo "All datasets dispatched — waiting for ${#RUNNING_PIDS[@]} remaining job(s)..."
    while [[ ${#RUNNING_PIDS[@]} -gt 0 ]]; do
        wait_one
    done
fi

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
echo "=============================="
echo "  Benchmark Complete"
echo "=============================="
echo "  Passed     : $PASS / $TOTAL"
echo "  Failed     : $FAIL / $TOTAL"
echo "  Total time : ${HOURS}h ${MINS}m ${SECS}s"
echo ""
echo "  Per-dataset times:"
for DATASET in "${DATASETS[@]}"; do
    if [[ -v "DATASET_ELAPSED[$DATASET]" ]]; then
        d_elapsed=${DATASET_ELAPSED[$DATASET]}
        d_mins=$(( d_elapsed / 60 ))
        d_secs=$(( d_elapsed % 60 ))
        printf "    %-22s  %dm %02ds\n" "$DATASET" "$d_mins" "$d_secs"
    fi
done

if [[ ${#FAILED_LIST[@]} -gt 0 ]]; then
    echo ""
    echo "  Failed datasets:"
    for d in "${FAILED_LIST[@]}"; do
        echo "    - $d"
    done
fi

echo ""
echo "  Results : $SCRIPT_DIR/$OUTPUT_DIR/$RESULT_SUBDIR/"
echo "  Logs    : $LOG_DIR/"
echo "=============================="
