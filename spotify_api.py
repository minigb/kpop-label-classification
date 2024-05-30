import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from omegaconf import OmegaConf
import logging
from tqdm.auto import tqdm
from pathlib import Path
from fuzzywuzzy import fuzz

# RELEASE_DIR = Path('releases')
ARTIST_DIR = Path('artists')
SONG_DIR = Path('songs')

# Configure logging
logging.basicConfig(filename='spotify_errors.log', level=logging.ERROR,
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
    for album in albums['items']:
        # Filter out albums where the artist is not the main artist
        if fuzz.ratio(album['artists'][0]['name'].lower(), artist_name.lower()) < 80:
            continue
        album_tracks = sp.album_tracks(album['id'])
        for track in album_tracks['items']:
            # Filter out tracks where the artist is not the main artist
            if track['artists'][0]['name'].lower() != artist_name.lower():
                continue
            # Extract year from release_date
            release_year = album['release_date'][:4]
            songs.append({
                'query_artist_name': query_artist_name,
                'song_title': track['name'],
                'album_name': album['name'],
                'release_year': release_year,
                'spotify_artist_name': artist_name
            })
    return songs

def save_to_csv(songs, filename):
    # Create a DataFrame from the list of songs
    df = pd.DataFrame(songs)
    if df.empty:
        logging.error("DataFrame is empty. No songs to save.")
        return
    # Sort the DataFrame by query_artist_name and release_year in ascending order
    df = df.sort_values(by=['query_artist_name', 'release_year'])
    # Check the DataFrame structure
    logging.info(f"DataFrame columns: {df.columns}")
    logging.info(f"DataFrame head: {df.head()}")
    # Save the DataFrame to a CSV file
    try:
        df.to_csv(filename, index=False)
    except Exception as e:
        logging.error(f"Error saving DataFrame to CSV: {e}")

def main():
    all_songs = []

    labels_with_song_list = SONG_DIR.glob('*.csv')
    for artist_fn in tqdm(list(ARTIST_DIR.glob('*.csv')), desc='Processing artists'):
        if artist_fn in labels_with_song_list:
            continue
        
        artist_df = pd.read_csv(artist_fn)
        artist_names = artist_df['artists'].unique()

        # Process each unique artist
        for query_artist_name in tqdm(artist_names):
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
