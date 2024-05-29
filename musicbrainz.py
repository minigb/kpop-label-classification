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
        'limit': 1500  # Increased limit to 1500
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
        for release in releases:
            release_group = release.get('release-group', {})
            if release_group.get('primary-type') not in ['Video']:
                release_data.append(release)
        offset += 1500
    return release_data

# Step 3: Save the Releases to a CSV file
def save_releases_to_csv(release_data, label_name):
    releases_info = []
    for release in release_data:
        releases_info.append({
            'release_id': release['id'],
            'title': release['title'],
            'status': release.get('status', ''),
            'release_date': release.get('date', ''),
            'country': release.get('country', ''),
        })
    
    df = pd.DataFrame(releases_info)
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
    # query_label_name = input("Enter the label name: ")
    query_label_name = 'ADOR'
    main(query_label_name)
