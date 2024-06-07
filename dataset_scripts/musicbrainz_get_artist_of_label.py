import requests
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
import logging
from fuzzywuzzy import fuzz
import datetime
import hydra

class MusicBrainzArtistByLabelCrawler:
    BASE_URL = "https://musicbrainz.org/ws/2/"
    LIMIT = 100  # Maximum limit per request set by MusicBrainz
    THRESHOLD = 80  # Fuzzy match threshold
    
    def __init__(self, config):
        self.save_dir = Path(config.data.artists_dir)
        self.log_file = f'musicbrainz_artists_per_label_errors_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'
        self._setup_logging()
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        if Path(self.log_file).exists():
            Path(self.log_file).unlink()
        logging.basicConfig(filename=self.log_file, level=logging.ERROR,
                            format='%(asctime)s:%(levelname)s:%(message)s')

    def get_label_id_and_name(self, label_name):
        try:
            url = f"{self.BASE_URL}label/"
            params = {
                'query': label_name,
                'fmt': 'json'
            }
            response = requests.get(url, params=params)
            data = response.json()
            if 'labels' in data:
                if len(data['labels']) == 1:
                    return data['labels'][0]['id'], data['labels'][0]['name']
                for label in data['labels']:
                    if label.get('country') in ['KR', 'XW']:
                        label_name_ratio = fuzz.ratio(label_name.lower().replace('entertainment', ''), label['name'].lower().replace('entertainment', ''))
                        if label_name_ratio > 80:
                            return label['id'], label['name']
            return None, None
        except Exception as e:
            logging.error(f"Error getting label ID for {label_name}: {e}")
            return None, None

    def get_releases(self, label_id, retrieved_label_name, query_label_name):
        try:
            url = f"{self.BASE_URL}release/"
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
                for release in tqdm(releases):
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
            url = f"{self.BASE_URL}release/{release_id}"
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
            df = pd.DataFrame(artist_data).drop_duplicates()
            df.to_csv(f'{self.save_dir}/{label_name}.csv', index=False)
        except Exception as e:
            logging.error(f"Error saving artists to CSV for label {query_label_name}: {e}")

    def get_artists_in_the_label(self, label_name):
        existing_files = list(self.save_dir.glob('*.csv'))
        if any(label_name in file.name for file in existing_files):
            df = pd.read_csv(f'{self.save_dir}/{label_name}.csv')
            if not df.empty:
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

    def run(self):
        for csv_file in tqdm(list(self.save_dir.glob('*.csv'))):
            label_name = csv_file.stem
            try:
                df = pd.read_csv(csv_file)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame(columns=['artists', 'artist_id'])
                df.to_csv(csv_file, index=False)
            if df.empty:
                print(f"Processing label: {label_name}")
                self.get_artists_in_the_label(label_name)


@hydra.main(config_path="../config", config_name='packed')
def main(config):
    crawler = MusicBrainzArtistByLabelCrawler(config)
    crawler.run()


if __name__ == '__main__':
    main()