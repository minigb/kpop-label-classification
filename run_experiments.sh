#!/bin/bash

# Array of model names to test
models=(
        # "Basic" # This is already done
        # Basic2Conv30s1seg
        # Basic3Conv30s1seg
        # Basic3Conv30s1seg_5year
        # Basic2Conv30s1seg_5year
        # Basic2Conv15s2seg_5year
        # Basic2Conv30s1seg_5year_16kHz
        Basic2Conv15s1seg_5year
        Basic2Conv15s1seg_5year_2chan
        )

# Loop through each model and run the training script
for model in "${models[@]}"
do
    echo "Running experiment with model: $model"
    python train.py model=$model
done
