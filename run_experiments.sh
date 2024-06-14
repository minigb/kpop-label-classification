#!/bin/bash

# Array of model names to test
models=(
        Basic2Conv15s1seg_5year_yearonly
        )

# Loop through each model and run the training script
for model in "${models[@]}"
do
    echo "Running experiment with model: $model"
    python train.py model=$model
done
