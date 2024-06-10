#!/bin/bash

# Array of model names to test
models=(
        # "Basic" # This is already done
        # Basic2Conv30s1seg
        # Basic3Conv30s1seg
        Basic3Conv30s1seg_5year
        )

# Loop through each model and run the training script
for model in "${models[@]}"
do
    echo "Running experiment with model: $model"
    python train.py model=$model
done
