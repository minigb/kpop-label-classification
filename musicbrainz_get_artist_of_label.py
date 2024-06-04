import requests
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
import logging
from fuzzywuzzy import fuzz
import datetime

class MusicBrainzLabelClassifier:
    def __init__(self):
        # Remove the existing log file if it exists
        now = datetime.datetime.now()
        log_file = f'musicbrainz_artists_per_label_errors_{now.strftime("%Y-%m-%d_%H-%M-%S")}.log'

        # Configure logging
        logging.basicConfig(filename=log_file, level=logging.ERROR,
                            format='%(asctime)s:%(levelname)s:%(message)s')

        self.LIMIT = 100  # Maximum limit per request set by MusicBrainz
        self.SAVE_DIR = Path('releases')
        if not self.SAVE_DIR.exists():
            self.SAVE_DIR.mkdir(parents=True, exist_ok=True)

    def get_label_id_and_name(self, label_name):
        try:
            url = "https://musicbrainz.org/ws/2/label/"
            params = {
                'query': label_name,
                'fmt': 'json'
            }
            response = requests.get(url, params=params)
            data = response.json()
            if 'labels' in data:
                for label in data['labels']:
                    if label.get('country') == 'KR':
                        label_name_ratio = fuzz.ratio(label_name.lower().replace('entertainment', ''), label['name'].lower().replace('entertainment', ''))
                        if label_name_ratio > 80:
                            return label['id'], label['name']
            return None, None
        except Exception as e:
            logging.error(f"Error getting label ID for {label_name}: {e}")
            return None, None

    def get_releases(self, label_id, retrieved_label_name, query_label_name):
        try:
            url = f"https://musicbrainz.org/ws/2/release/"
            params = {
                'label': label_id,
                'fmt': 'json',
                'limit': self.LIMIT
            }
            release_data = []
            offset = 0
            while True:
                params['offset'] = offset
                response = requests.get(url, params=params)
                data = response.json()
                releases = data.get('releases', [])
                if not releases:
                    break
                for release in tqdm(releases, desc=f"Processing releases for label {query_label_name}"):
                    release_id = release['id']
                    release_details = self.get_release_details(release_id, retrieved_label_name, query_label_name)
                    if release_details:
                        release_data.append(release_details)
                offset += self.LIMIT
            return release_data
        except Exception as e:
            logging.error(f"Error getting releases for label ID {label_id}: {e}")
            return []

    def get_release_details(self, release_id, retrieved_label_name, query_label_name):
        try:
            url = f"https://musicbrainz.org/ws/2/release/{release_id}"
            params = {
                'inc': 'artist-credits',
                'fmt': 'json'
            }
            response = requests.get(url, params=params)
            data = response.json()
            artist_details = []
            for artist_credit in data.get('artist-credit', []):
                artist_name = artist_credit['artist']['name']
                artist_id = artist_credit['artist']['id']
                artist_details.append({'artists': artist_name, 'artist_id': artist_id})
            return {
                'release_id': data['id'],
                'title': data['title'],
                'status': data.get('status', ''),
                'release_date': data.get('date', ''),
                'country': data.get('country', ''),
                'artists': artist_details,
                'retrieved_label_name': retrieved_label_name,
                'query_label_name': query_label_name
            }
        except Exception as e:
            logging.error(f"Error getting release details for release ID {release_id}: {e}")
            return None

    def save_artists_to_csv(self, release_data, query_label_name):
        try:
            artist_data = []
            for release in release_data:
                for artist in release['artists']:
                    artist_data.append(artist)
            label_name = f"{query_label_name.replace('/', '_')}"
            df = pd.DataFrame(artist_data)
            df.to_csv(f'{self.SAVE_DIR}/{label_name}.csv', index=False)
        except Exception as e:
            logging.error(f"Error saving artists to CSV for label {query_label_name}: {e}")

    def main(self, label_name):
        existing_files = list(self.SAVE_DIR.glob('*.csv'))
        if any(label_name in file.name for file in existing_files):
            print(f"Releases for {label_name} already saved to CSV")
            return

        label_id, retrieved_label_name = self.get_label_id_and_name(label_name)
        if not label_id:
            logging.error(f"Label not found for {label_name}")
            return

        release_data = self.get_releases(label_id, retrieved_label_name, label_name)
        if not release_data:
            logging.error(f"No releases found for {label_name}")
            return

        self.save_artists_to_csv(release_data, label_name)

if __name__ == '__main__':
    classifier = MusicBrainzLabelClassifier()
    fn = Path('kpop-dataset/song_list.csv')
    df = pd.read_csv(fn)
    labels = df['Label'].unique()
    for label_name in tqdm(labels, desc="Processing labels"):
        print(f"Processing label: {label_name}")
        classifier.main(label_name)