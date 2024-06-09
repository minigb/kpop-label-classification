#!/bin/bash

# Array of model names to test
models=(
        # "Basic" # This is already done
        "Basic3Convs15s1seg"
        "Basic3Convs15s2seg"
        "Basic3Convs30s1seg"
        "Basic3Convs30s2seg"
        )

# Loop through each model and run the training script
for model in "${models[@]}"
do
    echo "Running experiment with model: $model"
    python train.py model=$model
done
