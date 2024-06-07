#!/bin/bash

set -e
dir_name="dataset_scripts"

scripts=(
    # "musicbrainz_get_artist_of_label.py"
    # "musicbrainz_song.py"

    "musicbrainz_check_track_artist.py"
    "make_label_column.py"

    # be careful when running this
    "crawling_data.py"

    "annotate_label.py"
    "categorize_song_items.py"
)

for script in "${scripts[@]}"; 
do
    echo "Running $script"
    python "$dir_name"/"$script"
done

echo "All scripts have been executed successfully."