#!/bin/bash

# Array of model names to test
models=(
        # "Basic" # This is already done
        "BasicWith3Convs"
        "BasicWith15s1seg"
        "BasicWith15s2seg"
        "BasicWith30s2seg"
        )

# Loop through each model and run the training script
for model in "${models[@]}"
do
    echo "Running experiment with model: $model"
    python train.py model=$model
done
