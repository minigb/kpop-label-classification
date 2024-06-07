import pandas as pd
from pathlib import Path
from tqdm import tqdm
import hydra
from sklearn.model_selection import train_test_split

from utils import load_json

class LabelAnnotator:
    def __init__(self, config):
        self.song_list_csv_fn = Path(config.kpop_dataset.song_list_csv_fn)
        self.artists_per_label_dir = Path(config.data.artists_dir)
        self.recordings_per_artist_dir = Path(config.data.recordings_dir)
        self.artist_list_csv_fn = Path(config.kpop_dataset.artist_list_csv_fn)

        self.major_label_fn = Path(config.kpop_dataset.type.major_label_fn)
        self.column_name = config.column_name
        self.test_size = config.test_size
        self.random_seed = config.random_seed

        self.df = pd.read_csv(self.song_list_csv_fn)

    def run(self):
        self._match_label()
        self._annotate_is_major_label()
        self._check_columns()
        self._save_csv()

    def _match_label(self):
        label_list = []

        artist_label_df = pd.read_csv(self.artist_list_csv_fn)
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            artist_name = row[self.column_name.song.artist]
            artist_df = artist_label_df[artist_label_df[self.column_name.artist] == artist_name]
            assert not artist_df.empty

            if len(artist_df) == 1:
                label = artist_df.iloc[0][self.column_name.label]
                label_list.append(label)
                continue

            label= None
            release_date = str(row[self.column_name.date])
            for _, artist_df_row in artist_df.iterrows():
                start_date = artist_df_row[self.column_name.label_start]
                end_date = artist_df_row[self.column_name.label_end]

                assert not (pd.isnull(start_date) and pd.isnull(end_date))
                if pd.isnull(start_date):
                    start_date = '0'
                if pd.isnull(end_date):
                    end_date = '9'
                assert start_date < end_date
                
                if start_date <= release_date <= end_date:
                    label = artist_df_row[self.column_name.label]
                    break

            assert label is not None, f"Cannot find label for {artist_name} ({release_date})"
            label_list.append(label)
        self.df[self.column_name.label] = label_list


    def _annotate_is_major_label(self):
        self.df[self.column_name.is_major_label] = False
        label_dict = load_json(self.major_label_fn)
        for _, label_list in label_dict.items():
            for label_name in label_list:
                idxs = self.df[self.df[self.column_name.label] == label_name].index
                self.df.loc[idxs, self.column_name.is_major_label] = True
    def _check_columns(self):
        # check nan
        for column_name in [self.column_name.label]:
            assert self.df[column_name].notna().all(), f'Nan value in {column_name} column.'

    def _save_csv(self):
        self.df.to_csv(self.song_list_csv_fn, index=False)


@hydra.main(config_path="config", config_name="packed")
def main(config):
    annotator = LabelAnnotator(config)
    annotator.run()

if __name__ == "__main__":
    main()