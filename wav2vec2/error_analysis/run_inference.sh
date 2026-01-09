#!/bin/bash

# Exit immediately if a command fails
set -e

# Define languages
languages=("zu" "xh" "nbl" "ssw")

# Loop through each language
for lang in "${languages[@]}"; do
    echo "============================"
    echo "Starting Language: $lang"
    echo "============================"

    python infer_error_multilingual.py --language "$lang"

    echo "Finished Language: $lang"
    echo "Sleeping for 60 seconds before next..."
    sleep 60
done

echo "All multilingual languages completed."
