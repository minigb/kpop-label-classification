import pandas as pd
from pathlib import Path
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from utils import load_json

class DataCleaner:
    def __init__(self, config):
        self.recordings_dir = Path(config.data.recordings_dir)
        self.artists_dir = Path(config.data.artists_dir)
        self.removed_recordings_dir = Path(config.data.removed_recordings_dir)
        self.song_list_csv_fn = Path(config.dataset.song_list_csv_fn)
        self.excluding_keywords_for_song_title = load_json(config.exclude_keywords.song_title)
        self.excluding_keywords_for_song_info = load_json(config.exclude_keywords.song_info)

    def run(self):
        self._standardize(['title'])
        self._remove_rows_with_nan_of_these_columns(['track_artist', 'release_date', 'title'])
        self._remove_multiple_artists()
        self._check_artist_names()
        self._sort_by_columns(['title', 'release_date'])
        self._remove_duplicated_recording()
        self._remove_different_ver()
        self._remove_other_types()
        self._remove_empty_csv()
        self._make_total_song_list_csv()

    def _standardize(self, columns):
        for csv_file in tqdm(list(self.recordings_dir.glob('*.csv'))):
            df = pd.read_csv(csv_file)
            for column in columns:
                df[column] = df[column].apply(lambda x: x.replace('’', "'"))
                dash = '-'
                df[column] = df[column].apply(lambda x: x.replace('–', dash).replace('—', dash))
            df.to_csv(csv_file, index=False)

    def _remove_multiple_artists(self):
        artists_used = []
        for csv_file in tqdm(list(self.artists_dir.glob('*.csv'))):
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
        
        self.removed_recordings_dir.mkdir(parents=True, exist_ok=True)
        for artist_fn in tqdm(list(self.recordings_dir.glob('*.csv'))):
            artist_name = artist_fn.stem
            csv_fn = self.recordings_dir / f'{artist_name.replace("/", "_")}.csv'
            if not csv_fn.exists():
                continue
            if artist_name not in artists_used:
                csv_fn.rename(self.removed_recordings_dir / csv_fn.name)

    def _check_artist_names(self):
        def _refine_artist_name(artist_name):
            return artist_name.replace('.', '').lower()

        for csv_fn in tqdm(list(self.recordings_dir.glob('*.csv'))):
            df = pd.read_csv(csv_fn)
            idx_to_remove = []
            for idx, row in df.iterrows():
                query_artist = row['query_artist']
                first_artist = row['track_artist'].split(', ')[0]
                if _refine_artist_name(query_artist) != _refine_artist_name(first_artist):
                    idx_to_remove.append(idx)
            df = df.drop(idx_to_remove)
            df.to_csv(csv_fn, index=False)

    def _remove_empty_csv(self):
        for csv_fn in tqdm(list(self.recordings_dir.glob('*.csv'))):
            df = pd.read_csv(csv_fn)
            if df.empty:
                csv_fn.rename(self.removed_recordings_dir / csv_fn.name)

    def _remove_rows_with_nan_of_these_columns(self, columns):
        for csv_fn in tqdm(list(self.recordings_dir.glob('*.csv'))):
            df = pd.read_csv(csv_fn)
            df.dropna(subset=columns, inplace=True)
            df.to_csv(csv_fn, index=False)

    def _sort_by_columns(self, columns):
        for csv_fn in tqdm(list(self.recordings_dir.glob('*.csv'))):
            df = pd.read_csv(csv_fn)
            df_copy = df.copy()
            for col in columns:
                if df_copy[col].dtype == 'object':
                    df_copy[col] = df_copy[col].str.lower()
            sorted_df_copy = df_copy.sort_values(columns)
            sorted_df = df.loc[sorted_df_copy.index]
            sorted_df.to_csv(csv_fn, index=False)

    def _remove_duplicated_recording(self):
        for csv_fn in tqdm(list(self.recordings_dir.glob('*.csv'))):
            df = pd.read_csv(csv_fn)
            titles = df['title'].str.lower()
            duplicates = titles.duplicated(keep='first')
            df = df[~duplicates]
            df.to_csv(csv_fn, index=False)

    def _remove_different_ver(self):
        for csv_fn in tqdm(list(self.recordings_dir.glob('*.csv'))):
            df = pd.read_csv(csv_fn)
            idx_to_remove = []
            for idx, row in df.iterrows():
                title = row['title']
                for keyword in self.excluding_keywords_for_song_title:
                    if keyword.lower() in title.lower():
                        idx_to_remove.append(idx)
                        break
            df = df.drop(idx_to_remove)
            df.to_csv(csv_fn, index=False)

    def _remove_other_types(self):
        for csv_fn in tqdm(list(self.recordings_dir.glob('*.csv'))):
            df = pd.read_csv(csv_fn)
            idx_to_remove = []
            for idx, row in df.iterrows():
                row_str = ' '.join(map(str, row.values))
                for keyword in self.excluding_keywords_for_song_info:
                    if keyword.lower() in row_str.lower():
                        idx_to_remove.append(idx)
                        break
            df = df.drop(idx_to_remove)
            df.to_csv(csv_fn, index=False)

    def _make_total_song_list_csv(self):
        song_list = []
        for csv_fn in tqdm(list(self.recordings_dir.glob('*.csv'))):
            df = pd.read_csv(csv_fn)
            df['year'] = df['release_date'].apply(lambda x: int(x.split('-')[0]) if isinstance(x, str) else int(x))
            song_list.append(df)
        if song_list:
            song_list_df = pd.concat(song_list, ignore_index=True)
            song_list_df.to_csv(self.song_list_csv_fn, index=False)
        else:
            print("No data found to merge.")


@hydra.main(config_path='config', config_name='packed')
def main(config: DictConfig):
    processor = DataCleaner(config)
    processor.run()


if __name__ == '__main__':
    main()
