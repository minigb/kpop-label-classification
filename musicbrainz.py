import requests
import pandas as pd

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
        'limit': 1500  # Set limit to 100 to get more results per request
    }
    response = requests.get(url, params=params)
    data = response.json()
    releases = data.get('releases', [])
    release_ids = [release['id'] for release in releases]

    # Handle pagination
    while 'release-offset' in data:
        params['offset'] = data['release-offset'] + 100
        response = requests.get(url, params=params)
        data = response.json()
        releases = data.get('releases', [])
        release_ids.extend([release['id'] for release in releases])
        if not releases:
            break

    return release_ids

# Step 3: Get the Artists from the Releases
def get_artists_from_releases(release_ids):
    artists = set()
    for release_id in release_ids:
        url = f"https://musicbrainz.org/ws/2/release/{release_id}"
        params = {
            'inc': 'artist-credits',
            'fmt': 'json'
        }
        response = requests.get(url, params=params)
        data = response.json()
        if 'artist-credit' in data:
            for artist_credit in data['artist-credit']:
                artists.add(artist_credit['artist']['name'])
    return list(artists)

def save_to_csv(artists, label_name):
    # Create a DataFrame from the list of artists
    df = pd.DataFrame(artists, columns=['artist_name'])
    # Save the DataFrame to a CSV file
    df.to_csv(f'{label_name}_artists.csv', index=False)

def main(label_name):
    label_id = get_label_id(label_name)
    if not label_id:
        print("Label not found")
        return

    release_ids = get_releases(label_id)
    if not release_ids:
        print("No releases found for this label")
        return

    artists = get_artists_from_releases(release_ids)
    save_to_csv(artists, label_name)
    print(f"Saved {len(artists)} artists to {label_name}_artists.csv")

if __name__ == '__main__':
    # query_label_name = input("Enter the label name: ")
    query_label_name = "SM Entertainment"
    main(query_label_name)
