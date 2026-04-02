#!/bin/bash

# Number of epochs and batch size
EPOCHS=10
BATCH_SIZE=64

# Dataset list
DATASETS=(
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
    capture24
)

echo "Starting HarNet LOSO runs..."
echo "----------------------------------"

for DATASET in "${DATASETS[@]}"
do
    echo ""
    echo "=================================="
    echo "Running dataset: $DATASET"
    echo "=================================="

    python ./harnet_har.py \
        --dataset_name "$DATASET" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE

    echo "Finished dataset: $DATASET"
done

echo ""
echo "All runs completed."