#!/bin/bash

# ==============================
# HAR LOSO Batch Runner
# ==============================

SCRIPT=units_har.py
EPOCHS=10
BATCH=64
LR=1e-4

DATASETS=(
capture24
# daphnet
# dsads
# harsense
# harth
# hhar
# hugadb
# ku_har
# mhealth
# motion_sense
# opportunity
# realworld
# uci_har
# usc_had
# w_har
# wisdm_19_phone
# wisdm_19_watch
# wisdm
)

echo "Starting LOSO runs for ${#DATASETS[@]} datasets..."
echo "==============================================="

for ds in "${DATASETS[@]}"
do
    echo ""
    echo "==============================================="
    echo "Running dataset: $ds"
    echo "==============================================="

    python $SCRIPT \
        --dataset_name $ds \
        --epochs $EPOCHS \
        --batch_size $BATCH \
        --lr $LR

    echo "Finished: $ds"
done

echo ""
echo "All datasets completed."