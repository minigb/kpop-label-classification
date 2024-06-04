import requests
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
import logging
from fuzzywuzzy import fuzz
import hydra

# Remove the existing log file if it exists
log_file = 'musicbrainz_songs_errors.log'
if Path(log_file).exists():
    Path(log_file).unlink()

# Configure logging
logging.basicConfig(filename=log_file, level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

BASE_URL = "https://musicbrainz.org/ws/2/"
LIMIT = 100  # Maximum limit per request set by MusicBrainz
THRESHOLD = 80  # Fuzzy match threshold

# Step 1: Get the Artist ID and Name
def get_artist_id(query_artist_name):
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
                if 'country' in artist.keys() and artist['country'] not in ['KR']:
                    continue
                ratio = fuzz.ratio(query_artist_name.lower(), artist['name'].lower())
                if ratio > THRESHOLD:
                    return artist['id']
        return None
    except Exception as e:
        logging.error(f"Error getting artist ID for {query_artist_name}: {e}")
        return None

# Step 2: Get the Recordings of the Artist

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
    

def get_recordings(artist_id, query_artist_name):
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
                recording_info = get_recording_info(recording['id'])
                if not recording_info:
                    continue
                recording_info['query_artist'] = query_artist_name
                recordings.append(recording_info)
            offset += LIMIT
            save_recordings_to_csv(recordings, query_artist_name, save_dir) # save every time

        return recordings
    except Exception as e:
        logging.error(f"Error getting recordings for artist ID {artist_id}: {e}")
        return []

# Step 3: Save the Recordings to a CSV file
def save_recordings_to_csv(recordings, artist_name, save_dir):
    try:
        df = pd.DataFrame(recordings)
        df.to_csv(f'{save_dir}/{artist_name.replace("/", "_")}.csv', index=False)
    except Exception as e:
        logging.error(f"Error saving recordings to CSV for artist {artist_name}: {e}")

def process_artist(query_artist_name, artist_id):
    # check if the file exist
    if (save_dir / f'{query_artist_name.replace("/", "_")}.csv').exists():
        return
    
    print(f"Processing artist: {query_artist_name}")
    artist_id = artist_id if artist_id is not None else get_artist_id(query_artist_name)
    if not artist_id:
        logging.error(f"Artist not found for {query_artist_name}")
        return

    recordings = get_recordings(artist_id, query_artist_name)
    if not recordings:
        logging.error(f"Artist is found, but no recordings found for artist {query_artist_name}")
        return
    save_recordings_to_csv(recordings, query_artist_name, save_dir)

def get_unique_artists_from_csv(artists_dir):
    all_dataframes = []
    for csv_file in artists_dir.glob('*.csv'):
        df = pd.read_csv(csv_file)
        all_dataframes.append(df)
    combined_df = pd.concat(all_dataframes)
    # Leave unique artists
    unique_artists_df = combined_df.drop_duplicates(subset='artists', keep=False)
    unique_artists_df['artist_id'].fillna('')
    return unique_artists_df


@hydra.main(config_path='config', config_name='packed')
def main(config):
    artists_dir = Path(config.data.artists_dir)
    save_dir = Path(config.data.recordings_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    artists_df = get_unique_artists_from_csv(artists_dir)
    for _, row in tqdm(artists_df.iterrows(), desc="Processing artists", total=len(artists_df)):
        artist_name = row['artists']
        artist_id = row['artist_id'] if row['artist_id'] else None
        process_artist(artist_name, artist_id)

if __name__ == '__main__':
    main()