import requests
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
import logging
from fuzzywuzzy import fuzz

# Remove the existing log file if it exists
log_file = 'musicbrainz_songs_errors.log'

# Configure logging
logging.basicConfig(filename=log_file, level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

BASE_URL = "https://musicbrainz.org/ws/2/"
LIMIT = 100  # Maximum limit per request set by MusicBrainz
THRESHOLD = 80  # Fuzzy match threshold
ARTIST_DIR = Path('artists')  # Directory containing artist CSV files
SAVE_DIR = Path('recordings')
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Step 1: Get the Artist ID and Name
def get_artist_id_and_name(query_artist_name):
    try:
        url = f"{BASE_URL}artist/"
        params = {
            'query': query_artist_name,
            'fmt': 'json'
        }
        response = requests.get(url, params=params)
        data = response.json()
        if 'artists' in data and len(data['artists']) > 0:
            for artist in data['artists']:
                if artist['country'] not in ['KR', 'JP', 'CN', 'XW']:
                    continue
                ratio = fuzz.ratio(query_artist_name.lower(), artist['name'].lower())
                if ratio > THRESHOLD:
                    return artist['id'], artist['name']
        return None, None
    except Exception as e:
        logging.error(f"Error getting artist ID for {query_artist_name}: {e}")
        return None, None

# Step 2: Get the Recordings of the Artist
def get_recordings(artist_id, retrieved_artist_name, query_artist_name):
    try:
        url = f"{BASE_URL}recording/"
        params = {
            'artist': artist_id,
            'fmt': 'json',
            'limit': LIMIT
        }
        recordings = []
        offset = 0

        while True:
            params['offset'] = offset
            response = requests.get(url, params=params)
            data = response.json()
            recordings_data = data.get('recordings', [])
            if not recordings_data:
                break
            for recording in recordings_data:
                recordings.append({
                    'recording_id': recording['id'],
                    'title': recording['title'],
                    'release_date': recording.get('first-release-date', ''),
                    'query_artist_name': query_artist_name,
                    'retrieved_artist_name': retrieved_artist_name
                })
            offset += LIMIT

        return recordings
    except Exception as e:
        logging.error(f"Error getting recordings for artist ID {artist_id}: {e}")
        return []

# Step 3: Save the Recordings to a CSV file
def save_recordings_to_csv(recordings, artist_name):
    try:
        df = pd.DataFrame(recordings)
        df.to_csv(f'{SAVE_DIR}/{artist_name.replace("/", "_")}.csv', index=False)
    except Exception as e:
        logging.error(f"Error saving recordings to CSV for artist {artist_name}: {e}")

def process_artist(query_artist_name):
    artist_id, retrieved_artist_name = get_artist_id_and_name(query_artist_name)
    if not artist_id:
        logging.error(f"Artist not found for {query_artist_name}")
        return

    recordings = get_recordings(artist_id, retrieved_artist_name, query_artist_name)
    if not recordings:
        logging.error(f"No recordings found for artist {query_artist_name}")
        return

    save_recordings_to_csv(recordings, query_artist_name)

def get_unique_artists_from_csv():
    all_artists = set()
    for csv_file in ARTIST_DIR.glob('*.csv'):
        df = pd.read_csv(csv_file)
        artists = df['artists'].unique()
        all_artists.update(artists)
    return list(all_artists)

if __name__ == '__main__':
    artists = get_unique_artists_from_csv()
    for artist_name in tqdm(artists, desc="Processing artists"):
        print(f"Processing artist: {artist_name}")
        process_artist(artist_name)
