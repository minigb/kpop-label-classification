import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from omegaconf import OmegaConf
import logging
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(filename='errors.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Load the Spotify API credentials from the YAML file
spotify_conf = OmegaConf.load('spotify_keys.yaml')
client_id = spotify_conf.client_id
client_secret = spotify_conf.client_secret

# Authenticate with Spotify
auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

def get_artist_id_and_name(artist_name):
    # Search for the artist by name
    result = sp.search(q=artist_name, type='artist')
    # Get the artist ID and name from the search results
    artist = result['artists']['items'][0]
    artist_id = artist['id']
    artist_name = artist['name']
    return artist_id, artist_name

def get_all_songs(artist_id, artist_name, query_artist_name):
    songs = []
    # Get all albums and singles by the artist
    albums = sp.artist_albums(artist_id, album_type='album,single', limit=50)
    for album in tqdm(albums['items']):
        # Filter out albums where the artist is not the main artist
        if album['artists'][0]['name'].lower() != artist_name.lower():
            continue
        album_tracks = sp.album_tracks(album['id'])
        for track in album_tracks['items']:
            # Filter out tracks where the artist is not the main artist
            if track['artists'][0]['name'].lower() != artist_name.lower():
                continue
            songs.append({
                'query_artist_name': query_artist_name,
                'song_title': track['name'],
                'album_name': album['name'],
                'release_date': album['release_date'],
                'spotify_artist_name': artist_name
            })
    return songs

def save_to_csv(songs, filename):
    # Create a DataFrame from the list of songs
    df = pd.DataFrame(songs)
    if df.empty:
        logging.error("DataFrame is empty. No songs to save.")
        return
    # Convert release_date to datetime for sorting
    try:
        df['release_date'] = pd.to_datetime(df['release_date'])
    except Exception as e:
        logging.error(f"Error converting release_date to datetime: {e}")
        return
    # Sort the DataFrame by release_date in ascending order
    df = df.sort_values(by='release_date')
    # Check the DataFrame structure
    logging.info(f"DataFrame columns: {df.columns}")
    logging.info(f"DataFrame head: {df.head()}")
    # Save the DataFrame to a CSV file
    try:
        df.to_csv(filename, index=False)
    except Exception as e:
        logging.error(f"Error saving DataFrame to CSV: {e}")

def main():
    # Read the song_list.csv file
    try:
        song_list_df = pd.read_csv('song_list.csv')
    except Exception as e:
        logging.error(f"Error reading song_list.csv: {e}")
        return

    # Get unique combinations of Label and Artist
    unique_artists = song_list_df[['Label', 'Artist']].drop_duplicates()

    all_songs = []

    # Process each unique artist
    for _, row in tqdm(unique_artists.iterrows(), total=len(unique_artists)):
        query_artist_name = row['Artist']
        try:
            # Get the artist ID and the Spotify artist name for the given artist name
            artist_id, spotify_artist_name = get_artist_id_and_name(query_artist_name)
            # Get all songs by the artist
            songs = get_all_songs(artist_id, spotify_artist_name, query_artist_name)
            all_songs.extend(songs)
        except Exception as e:
            logging.error(f"Error processing artist {query_artist_name}: {e}")

        # Save all songs to a single CSV file
        save_to_csv(all_songs, 'all_songs.csv')
    print(f"Saved all songs to all_songs.csv")

if __name__ == '__main__':
    main()
