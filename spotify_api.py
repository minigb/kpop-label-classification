import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from omegaconf import OmegaConf

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
        if album['artists'][0]['name'].lower() != artist_name.lower():
            continue
        album_tracks = sp.album_tracks(album['id'])
        for track in album_tracks['items']:
            # Filter out tracks where the artist is not the main artist
            if track['artists'][0]['name'].lower() != artist_name.lower():
                continue
            songs.append({
                'song_title': track['name'],
                'album_name': album['name'],
                'release_date': album['release_date'],
                'artist_name': artist_name,
                'query_artist_name': query_artist_name,
            })
    return songs

def save_to_csv(songs, artist_name):
    # Create a DataFrame from the list of songs
    df = pd.DataFrame(songs)
    # Convert release_date to datetime for sorting
    df['release_date'] = pd.to_datetime(df['release_date'])
    # Sort the DataFrame by release_date in ascending order
    df = df.sort_values(by='release_date')
    # Save the DataFrame to a CSV file
    df.to_csv(f'{artist_name}_songs.csv', index=False)

def main(query_artist_name):
    # Get the artist ID and the Spotify artist name for the given artist name
    artist_id, spotify_artist_name = get_artist_id_and_name(query_artist_name)
    # Get all songs by the artist
    songs = get_all_songs(artist_id, spotify_artist_name, query_artist_name)
    # Save the songs to a CSV file
    save_to_csv(songs, spotify_artist_name)
    print(f"Saved {len(songs)} songs to {spotify_artist_name}_songs.csv")

if __name__ == '__main__':
    query_artist_name = input("Enter the artist name: ")
    main(query_artist_name)
