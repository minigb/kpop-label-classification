import pandas as pd
from pathlib import Path
from tqdm import tqdm

RECORDINGS_DIR = Path('recordings')
RECORDINGS_NOT_USED_DIR = Path('recordings_not_used')
ARTIST_DIR = Path('artists')


def standardize_apostrophes(columns):
    for csv_file in tqdm(list(RECORDINGS_DIR.glob('*.csv'))):
        df = pd.read_csv(csv_file)
        for column in columns:
            df[column] = df[column].apply(lambda x: x.replace('’', "'"))
        df.to_csv(csv_file, index=False)


def remove_multiple_artists():
    artists_not_used = []
    for csv_file in tqdm(list(ARTIST_DIR.glob('*.csv'))):
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
    for csv_fn in tqdm(list(RECORDINGS_DIR.glob('*.csv'))):
        df = pd.read_csv(csv_fn)
        if df.empty:
            csv_fn.rename(RECORDINGS_NOT_USED_DIR / csv_fn.name)


def remove_rows_with_nan_of_these_columns(columns):
    for csv_fn in RECORDINGS_DIR.glob('*.csv'):
        df = pd.read_csv(csv_fn)
        idx_to_remove = []
        for idx, row in df.iterrows():
            for column in columns:
                if pd.isna(row[column]):
                    idx_to_remove.append(idx)
                    break
        df = df.drop(idx_to_remove)
        df.to_csv(csv_fn, index=False)


def sort_by_columns(columns):
    for csv_fn in RECORDINGS_DIR.glob('*.csv'):
        df = pd.read_csv(csv_fn)
        df = df.sort_values(columns)
        df.to_csv(csv_fn, index=False)


def remove_duplicated_recording():
    for csv_fn in tqdm(list(RECORDINGS_DIR.glob('*.csv'))):
        df = pd.read_csv(csv_fn)
        title_set = set()
        idx_to_remove = []
        for idx, row in df.iterrows():
            title = row['title']
            if title in title_set:
                continue
            
            # Check recordings with this title
            title_set.add(title)
            start_idx = last_idx = idx + 1
            for idx2, row2 in df[start_idx:].iterrows():
                if not title in row2['title']:
                    last_idx = idx2
                    break
            idx_to_remove.extend(list(range(start_idx, last_idx)))

        df = df.drop(idx_to_remove)
        df.to_csv(csv_fn, index=False)


if __name__ == '__main__':
    standardize_apostrophes(['title'])
    # remove_rows_with_nan_of_these_columns(['track_artist', 'release_date', 'title'])
    # remove_multiple_artists()
    # check_artist_names()
    # sort_by_columns(['title', 'release_date'])
    # remove_duplicated_recording()

    remove_empty_csv()