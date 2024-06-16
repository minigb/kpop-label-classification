#!/bin/bash

# Array of model names to test
models=(
        Basic_labelonly
        Basic15s_5year
        )

# Loop through each model and run the training script
for model in "${models[@]}"
do
    echo "Running experiment with model: $model"
    python train.py model=$model
done
