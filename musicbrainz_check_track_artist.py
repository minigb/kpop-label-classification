import pandas as pd
from pathlib import Path
from tqdm import tqdm

RECORDINGS_DIR = Path('recordings')
RECORDINGS_NOT_USED_DIR = Path('recordings_not_used')
ARTIST_DIR = Path('artists')

def remove_multiple_artists():
    artists_not_used = []
    for csv_file in ARTIST_DIR.glob('*.csv'):
        df = pd.read_csv(csv_file)
        artists = df['artists']
        artists_idx_to_remove = []
        for idx, artist in enumerate(artists):
            if ', ' in artist:
                artists_idx_to_remove.append(idx)
                artists_not_used.append(artist)
        df = df.drop(artists_idx_to_remove)
        df.to_csv(csv_file, index=False)
    
    RECORDINGS_NOT_USED_DIR.mkdir(exist_ok=True)
    for artist_name in tqdm(artists_not_used):
        csv_fn = RECORDINGS_DIR / f'{artist_name.replace("/", "_")}.csv'
        if not csv_fn.exists():
            continue
        csv_fn.rename(RECORDINGS_NOT_USED_DIR / csv_fn.name)


def check_artist_names():
    def _refine_artist_name(artist_name):
        artist_name = artist_name.replace('.', '').lower()
        return artist_name

    for csv_fn in tqdm(list(RECORDINGS_DIR.glob('*.csv'))):
        df = pd.read_csv(csv_fn)
        idx_to_remove = []
        for idx, row in df.iterrows():
            query_artist = row['query_artist']
            first_artist = row['track_artist'].split(', ')[0]
            if _refine_artist_name(query_artist) != _refine_artist_name(first_artist):
                idx_to_remove.append(idx)

        df = df.drop(idx_to_remove)
        df.to_csv(csv_fn, index=False)


def remove_empty_csv():
    for csv_fn in RECORDINGS_DIR.glob('*.csv'):
        df = pd.read_csv(csv_fn)
        if df.empty:
            csv_fn.rename(RECORDINGS_NOT_USED_DIR / csv_fn.name)


if __name__ == '__main__':
    remove_multiple_artists()
    check_artist_names()
    remove_empty_csv()