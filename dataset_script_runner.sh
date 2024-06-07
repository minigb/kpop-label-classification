#!/bin/bash

set -e

# Set the directory name where the Python scripts are located
dir_name="dataset_scripts"

# List of Python scripts to run
scripts=(
    # "musicbrainz_get_artist_of_label.py"
    # "musicbrainz_song.py"

    "musicbrainz_check_track_artist.py"
    "make_label_column.py"

    # Be careful when running this
    "crawling_data.py"

    "annotate_label.py"
    "categorize_song_items.py"
)

# Iterate over the scripts and run each one
for script in "${scripts[@]}"; 
do
    echo "Running $script"
    python "$dir_name/$script"
done

echo "All scripts have been executed successfully."