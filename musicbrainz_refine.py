import csv
import requests
import logging
from tqdm import tqdm
import pandas as pd
from pathlib import Path

# Set up logging
logging.basicConfig(filename='artist_update.log', level=logging.INFO)

# Define directories
ARTIST_LIST_DIR = 'artists'
ARTIST_RECORDING_DIR = 'recordings_per_artist'

def get_recording_info(recording_id):
    try:
        url = f"https://musicbrainz.org/ws/2/recording/{recording_id}"
        params = {
            'inc': 'artists+releases',
            'fmt': 'json'
        }
        response = requests.get(url, params=params)
        recording = response.json()
        
        title = recording.get('title', '')
        length = recording.get('length', '')
        track_artist = ', '.join(artist['artist']['name'] for artist in recording.get('artist-credit', []))
        release = recording['releases'][0] if recording.get('releases') else {}
        release_title = release.get('title', '')
        release_artist = ', '.join(artist['artist']['name'] for artist in release.get('artist-credit', []))
        release_group_type = release.get('release-group', {}).get('type', '')
        country = release.get('country', '')
        release_date = release.get('date', '')
        label = ', '.join(label['label']['name'] for label in release.get('label-info', []))

        return {
            'title': title,
            'length': length,
            'track_artist': track_artist,
            'release_title': release_title,
            'release_artist': release_artist,
            'release_group_type': release_group_type,
            'country': country,
            'release_date': release_date,
            'label': label
        }
    except Exception as e:
        logging.error(f"Error fetching data for recording {recording_id}: {e}")
        return {}

# Get all the list of artist names from all the csv files
artist_names = set()

for file_name in Path(ARTIST_LIST_DIR).rglob('*.csv'):
    df = pd.read_csv(file_name)
    artist_names.update(df['artists'])

revised_artists = set()

# Process each artist
for artist_name in tqdm(artist_names, desc="Processing artists"):
    artist_file = Path(ARTIST_RECORDING_DIR) / f'{artist_name}.csv'
    if not artist_file.exists():
        logging.info(f'File not found for artist: {artist_name}')
        continue
    revised_artists.add(artist_name)

    df = pd.read_csv(artist_file)

    # Get new data from MusicBrainz API
    updated_records = []
    for _, record in df.iterrows():
        recording_id = record['recording_id']
        recording_info = get_recording_info(recording_id)
        if recording_info:
            updated_record = {
                'recording_id': recording_id,
                'title': recording_info.get('title', ''),
                'length': recording_info.get('length', ''),
                'track_artist': recording_info.get('track_artist', ''),
                'release_title': recording_info.get('release_title', ''),
                'release_artist': recording_info.get('release_artist', ''),
                'release_group_type': recording_info.get('release_group_type', ''),
                'country': recording_info.get('country', ''),
                'release_date': recording_info.get('release_date', ''),
                'label': recording_info.get('label', ''),
                'query_artist_name': record['query_artist_name'],
                'retrieved_artist_name': record['retrieved_artist_name']
            }
            updated_records.append(updated_record)

    # Save the updated records to the CSV file
    updated_df = pd.DataFrame(updated_records)
    updated_df.to_csv(artist_file, index=False)

# Remove unrevised artist CSV files
for file_name in Path(ARTIST_RECORDING_DIR).glob('*.csv'):
    artist_name = file_name.stem
    if artist_name not in revised_artists:
        file_name.unlink()
        logging.info(f'Removed unrevised artist file: {file_name}')
