#!/bin/bash

# Step 1: Run musicbrainz_song.py
python musicbrainz_song.py

# Step 2: Run musicbrainz_check_track_artist.py
python musicbrainz_check_track_artist.py

# Step 3: Run crawling_data.py with specified arguments
python crawling_data.py --input_csv kpop-dataset/song_list.csv --save_csv_name kpop-dataset/csv/kpop --save_audio_dir audio

echo "All scripts have been executed successfully."
