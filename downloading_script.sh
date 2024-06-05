#!/bin/bash

set -e

scripts=(
    # "musicbrainz_get_artist_of_label.py"
    # "musicbrainz_song.py"
    # "musicbrainz_check_track_artist.py"
    "make_label_column.py"
    "annotate_label.py"
    # "crawling_data.py"
)

for script in "${scripts[@]}"; 
do
    echo "Running $script"
    python "$script"
done

echo "All scripts have been executed successfully."