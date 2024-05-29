import requests
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
import logging

logging.basicConfig(filename='musicbrainz_errors.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

LIMIT = 1500

# Step 1: Get the Label ID
def get_label_id(label_name):
    url = "https://musicbrainz.org/ws/2/label/"
    params = {
        'query': label_name,
        'fmt': 'json'
    }
    response = requests.get(url, params=params)
    data = response.json()
    if 'labels' in data and len(data['labels']) > 0:
        return data['labels'][0]['id']
    return None

# Step 2: Get the Releases of the Label
def get_releases(label_id):
    url = f"https://musicbrainz.org/ws/2/release/"
    params = {
        'label': label_id,
        'fmt': 'json',
        'limit': LIMIT
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
            release_details = get_release_details(release_id)
            if release_details:
                release_data.append(release_details)
        offset += LIMIT
    return release_data

# Step 3: Get the detailed information of each release
def get_release_details(release_id):
    url = f"https://musicbrainz.org/ws/2/release/{release_id}"
    params = {
        'inc': 'artist-credits',
        'fmt': 'json'
    }
    response = requests.get(url, params=params)
    data = response.json()
    artist_names = [artist_credit['artist']['name'] for artist_credit in data.get('artist-credit', [])]
    return {
        'release_id': data['id'],
        'title': data['title'],
        'status': data.get('status', ''),
        'release_date': data.get('date', ''),
        'country': data.get('country', ''),
        'artists': ', '.join(artist_names)
    }

# Step 4: Save the Releases to a CSV file
def save_releases_to_csv(save_dir, release_data, label_name):
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir()

    df = pd.DataFrame(release_data)
    df.to_csv(f'{save_dir}/{label_name}_releases.csv', index=False)

def main(label_name, save_dir='releases'):
    label_id = get_label_id(label_name)
    if not label_id:
        logging.error("Label not found")
        return

    release_data = get_releases(label_id)
    if not release_data:
        logging.error(f"No releases found for {label_name}")
        return

    save_releases_to_csv(save_dir, release_data, label_name)

if __name__ == '__main__':
    labels = pd.read_csv('unique_labels.csv')['Label']
    for label_name in tqdm(labels):
        print(f"Processing label: {label_name}")
        main(label_name)
