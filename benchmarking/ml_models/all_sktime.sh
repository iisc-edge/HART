#!/bin/bash

# ============================================================
# Run LOSO sktime baseline on all HAR datasets
# ============================================================

DATASETS=(
    # capture24
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

OUTPUT_DIR="results_person"

echo "Starting sktime LOSO evaluation on all datasets..."
echo "Results directory: ${OUTPUT_DIR}"
echo "----------------------------------------------------"

for DATASET in "${DATASETS[@]}"
do
    echo ""
    echo "Running dataset: ${DATASET}"
    echo "----------------------------------------------------"

    python ml_sktime_person.py \
        --dataset_name "${DATASET}" \
        --output_dir "${OUTPUT_DIR}"

    echo "Finished dataset: ${DATASET}"
    echo "----------------------------------------------------"
done

echo ""
echo "All datasets completed."