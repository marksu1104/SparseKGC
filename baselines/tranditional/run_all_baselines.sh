#!/bin/bash

# Configuration
GPU=0
EPOCHS=500
BATCH=256
PATIENCE=25
EVAL_FREQ=1
EMB_DIM=200
LR=0.001

mkdir -p logs

DATASETS=("WD-singer" "FB15K-237-10" "WN18RR" "FB15K-237-20" "FB15K-237-50" "NELL23K" "FB15K-237")

for DATASET in "${DATASETS[@]}"; do
    DATA_PATH="../../datasets/$DATASET"

    echo "================================================"
    echo "Starting Baseline Experiments for $DATASET..."
    echo "Logs will be saved to 'logs/' directory and displayed in terminal."
    echo "Unified Parameters: Emb_Dim=$EMB_DIM, LR=$LR, Batch=$BATCH"
    echo "================================================"

    # 針對 WN18RR TransE 等大型網絡動態調整 Batch，防止 OOM
    if [ "$DATASET" == "WN18RR" ]; then
        TRANSE_BATCH=128
    else
        TRANSE_BATCH=$BATCH
    fi

    # 1. TransE
    echo "------------------------------------------------"
    echo "Running TransE for $DATASET with Batch Size $TRANSE_BATCH..."
    python main.py --model TransE --dataset $DATASET --data_path $DATA_PATH --gpu $GPU --max_epochs $EPOCHS --batch_size $TRANSE_BATCH --patience $PATIENCE --eval_freq $EVAL_FREQ --emb_dim $EMB_DIM --lr $LR --margin 9.0 2>&1 | tee logs/TransE_$DATASET.log

    # 2. DistMult
    echo "------------------------------------------------"
    echo "Running DistMult for $DATASET..."
    python main.py --model DistMult --dataset $DATASET --data_path $DATA_PATH --gpu $GPU --max_epochs $EPOCHS --batch_size $BATCH --patience $PATIENCE --eval_freq $EVAL_FREQ --emb_dim $EMB_DIM --lr $LR 2>&1 | tee logs/DistMult_$DATASET.log

    # 3. ComplEx (Added back)
    echo "------------------------------------------------"
    echo "Running ComplEx for $DATASET..."
    python main.py --model ComplEx --dataset $DATASET --data_path $DATA_PATH --gpu $GPU --max_epochs $EPOCHS --batch_size $BATCH --patience $PATIENCE --eval_freq $EVAL_FREQ --emb_dim $EMB_DIM --lr $LR 2>&1 | tee logs/ComplEx_$DATASET.log

    # 4. ConvE
    echo "------------------------------------------------"
    echo "Running ConvE for $DATASET..."
    python main.py --model ConvE --dataset $DATASET --data_path $DATA_PATH --gpu $GPU --max_epochs $EPOCHS --batch_size $BATCH --patience $PATIENCE --eval_freq $EVAL_FREQ --emb_dim $EMB_DIM --lr $LR 2>&1 | tee logs/ConvE_$DATASET.log

    # 5. TuckER
    echo "------------------------------------------------"
    echo "Running TuckER for $DATASET..."
    python main.py --model TuckER --dataset $DATASET --data_path $DATA_PATH --gpu $GPU --max_epochs $EPOCHS --batch_size $BATCH --patience $PATIENCE --eval_freq $EVAL_FREQ --emb_dim $EMB_DIM --lr $LR 2>&1 | tee logs/TuckER_$DATASET.log

    # 6. RotatE
    echo "------------------------------------------------"
    echo "Running RotatE for $DATASET..."
    python main.py --model RotatE --dataset $DATASET --data_path $DATA_PATH --gpu $GPU --max_epochs $EPOCHS --batch_size $BATCH --patience $PATIENCE --eval_freq $EVAL_FREQ --emb_dim $EMB_DIM --lr $LR --margin 9.0 2>&1 | tee logs/RotatE_$DATASET.log

done

echo "All baseline experiments completed. Checking logs directory and aggregating results..."
python aggregate_traditional_results.py
