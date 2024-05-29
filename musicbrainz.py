import requests
import pandas as pd
from tqdm.auto import tqdm

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
        'limit': 1500  # Limit set to 1500 for each request
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
        offset += 1500
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
def save_releases_to_csv(release_data, label_name):
    df = pd.DataFrame(release_data)
    df.to_csv(f'{label_name}_releases.csv', index=False)

def main(label_name):
    label_id = get_label_id(label_name)
    if not label_id:
        print("Label not found")
        return

    release_data = get_releases(label_id)
    if not release_data:
        print("No releases found for this label")
        return

    save_releases_to_csv(release_data, label_name)
    print(f"Saved {len(release_data)} releases to {label_name}_releases.csv")

if __name__ == '__main__':
    query_label_name = input("Enter the label name: ")
    main(query_label_name)
