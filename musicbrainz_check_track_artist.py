import pandas as pd
from pathlib import Path
from tqdm import tqdm

RECORDINGS_DIR = Path('recordings_per_artist')
RECORDINGS_NOT_USED_DIR = Path('recordings_not_used')
ARTIST_DIR = Path('artists_per_label')


def standardize(columns):
    for csv_file in tqdm(list(RECORDINGS_DIR.glob('*.csv'))):
        df = pd.read_csv(csv_file)
        for column in columns:
            df[column] = df[column].apply(lambda x: x.replace('’', "'"))
            df[column] = df[column].apply(lambda x: x.replace('‘', "'"))
        df.to_csv(csv_file, index=False)


def remove_multiple_artists():
    artists_used = []
    for csv_file in tqdm(list(ARTIST_DIR.glob('*.csv'))):
        df = pd.read_csv(csv_file)
        artists = df['artists']
        artists_idx_to_remove = []
        for idx, artist in enumerate(artists):
            if ', ' in artist:
                artists_idx_to_remove.append(idx)
            else:
                artists_used.append(artist)
        df = df.drop(artists_idx_to_remove)
        df.to_csv(csv_file, index=False)
    
    RECORDINGS_NOT_USED_DIR.mkdir(exist_ok=True)
    for artist_fn in tqdm(list(RECORDINGS_DIR.glob('*.csv'))):
        artist_name = artist_fn.stem
        csv_fn = RECORDINGS_DIR / f'{artist_name.replace("/", "_")}.csv'
        if not csv_fn.exists():
            continue
        if artist_name not in artists_used:
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


# def filter_by_country(country_list):
#     for csv_fn in tqdm(list(RECORDINGS_DIR.glob('*.csv'))):
#         df = pd.read_csv(csv_fn)
#         idx_to_remove = []
#         for idx, row in df.iterrows():
#             if row['country'] and row['country'] not in country_list:
#                 idx_to_remove.append(idx)
#         df = df.drop(idx_to_remove)
#         df.to_csv(csv_fn, index=False)


def sort_by_columns(columns):
    for csv_fn in tqdm(list(RECORDINGS_DIR.glob('*.csv'))):
        df = pd.read_csv(csv_fn)
        
        # Create a copy of the DataFrame and lowercase the specified columns
        df_copy = df.copy()
        for col in columns:
            if df_copy[col].dtype == 'object':
                df_copy[col] = df_copy[col].str.lower()
        
        # Sort the copy of the DataFrame
        sorted_df_copy = df_copy.sort_values(columns)
        
        # Reorder the original DataFrame using the sorted indices
        sorted_df = df.loc[sorted_df_copy.index]
        
        # Save the sorted DataFrame back to the CSV file
        sorted_df.to_csv(csv_fn, index=False)


def remove_duplicated_recording():
    for csv_fn in tqdm(list(RECORDINGS_DIR.glob('*.csv'))):
        df = pd.read_csv(csv_fn)

        idx_to_remove = []
        for idx, row in df.iterrows():
            if idx in idx_to_remove:
                continue
            title = row['title']

            # Check recordings with this title
            for idx2, row2 in df[idx + 1:].iterrows():
                if title.lower() in row2['title'].lower():
                    idx_to_remove.append(idx2)

        df = df.drop(idx_to_remove)
        df.to_csv(csv_fn, index=False)


def remove_different_ver(keywords = []):
    for csv_fn in tqdm(list(RECORDINGS_DIR.glob('*.csv'))):
        df = pd.read_csv(csv_fn)
        idx_to_remove = []
        for idx, row in df.iterrows():
            # check title
            title = row['title']
            for keyword in keywords:
                if keyword.lower() in title.lower():
                    idx_to_remove.append(idx)
                    break
        df = df.drop(idx_to_remove)
        df.to_csv(csv_fn, index=False)

def remove_other_types(keywords=[]):
    for csv_fn in tqdm(list(RECORDINGS_DIR.glob('*.csv'))):
        df = pd.read_csv(csv_fn)
        idx_to_remove = []
        for idx, row in df.iterrows():
            row_str = ' '.join(map(str, row.values))  # Convert all values to strings and join them
            for keyword in keywords:
                if keyword.lower() in row_str.lower():
                    idx_to_remove.append(idx)
                    break
        
        df = df.drop(idx_to_remove)
        df.to_csv(csv_fn, index=False)


def make_total_song_list_csv():
    song_list = []
    
    for csv_fn in tqdm(list(RECORDINGS_DIR.glob('*.csv'))):
        df = pd.read_csv(csv_fn)
        year_list = [int(date.split('-')[0]) if isinstance(date, str) else int(date) for date in df['release_date']]
        df['year'] = year_list
        song_list.append(df)
    
    if song_list:
        song_list_df = pd.concat(song_list, ignore_index=True)
        song_list_df.to_csv('kpop-dataset/song_list.csv', index=False)
    else:
        print("No data found to merge.")


if __name__ == '__main__':
    standardize(['title'])
    remove_rows_with_nan_of_these_columns(['track_artist', 'release_date', 'title'])
    remove_multiple_artists()
    check_artist_names()
    sort_by_columns(['title', 'release_date'])
    remove_duplicated_recording()
    remove_different_ver(['ver.', 'version', 'instrumental', 'inst.', 'remix', 'music video', \
                          'official mv', '(live)', '(Rearranged)', '(performance', 'm/v', 'teaser', \
                            '(ENG.)', 'TV', 'iKON SCHOOL'])
    remove_other_types(['Making of', 'ARENA TOUR', 'WORLD TOUR', 'DOME TOUR', 'LIVE TOUR', \
                         'Documentary of', 'behind the scenes', \
                        'FANCLUB EVENT', '2NE1 in Philippines',\
                        'Japan', 'Asia Promotion', 'Jacket Shooting Making', 'MAKING FILM', 'READY TO BE with', 'SPECIAL VIDEO',\
                        'Making Video',  'mv behind', 'teaser behind', 'Recording Making Movie',\
                        'LIVESTREAM CONCERT', 'DEBUT SHOWCASE', 'SPECIAL EDITION', 'THE LIVE', \
                        'BEST HIT MEGA BLEND', 'DOCUMENT FILM'])
    remove_empty_csv()

    make_total_song_list_csv()