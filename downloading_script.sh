#!/bin/bash

set -e

python musicbrainz_get_artist_of_label.py
python musicbrainz_song.py
# python musicbrainz_check_track_artist.py
# python crawling_data.py --input_csv kpop-dataset/song_list.csv --save_csv_name kpop-dataset/csv/kpop --save_audio_dir audio

echo "All scripts have been executed successfully."
