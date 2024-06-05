import pandas as pd
from pathlib import Path
import hydra

from utils import load_json

class LabelAnnotator:
    def __init__(self, config):
        self.artists_per_label_dir = Path(config.data.artists_dir)
        self.recordings_per_artist_dir = Path(config.data.recordings_dir)
        self.artist_list_csv_fn = Path(config.kpop_dataset.artist_list_csv_fn)
        # self.song_list_csv_fn = Path(config.kpop_dataset.song_list_csv_fn)
        self.song_list_csv_fn = Path('song_list_tmp.csv')
        self.case_study_artist_json_fn = Path(config.kpop_dataset.type.case_study.artist_fn)
        self.case_study_keyword_json_fn = Path(config.kpop_dataset.type.case_study.artist_fn)
        self.column_name = config.column_name

        self.df = pd.read_csv(self.song_list_csv_fn)

    def run(self):
        # self._create_empty_columns()
        self._match_label()
        self._save_csv()
        pass

    # def _create_empty_columns(self):
    #     self.df = pd.read_csv(self.song_list_csv_fn)
    #     for column_name in [self.column_name.label, self.column_name.is_train, self.column_name.is_test, self.column_name.is_case_study]:
    #         if column_name not in self.df.columns:
    #             self.df[column_name] = None

    def _match_label(self):
        label_list = []

        artist_label_df = pd.read_csv(self.artist_list_csv_fn)
        for _, row in self.df.iterrows():
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
                pick_this = False
                if pd.isnull(start_date):
                    if release_date <= end_date:
                        pick_this = True
                elif pd.isnull(end_date):
                    if start_date <= release_date:
                        pick_this = True
                else:
                    if start_date <= release_date <= end_date:
                        pick_this = True
                if pick_this:
                    label = artist_df_row[self.column_name.label]
                    break

            # assert label is not None, f"Cannot find label for {artist_name} ({release_date})"
            if label is None:
                with open('log.txt', 'a') as f:
                    f.write(f"Cannot find label for {artist_name} ({release_date})\n")
            label_list.append(label)
        self.df[self.column_name.label] = label_list

    def _annotate_case_study(self):
        artist_list = load_json(self.case_study_artist_json_fn)
        keyword_list = load_json(self.case_study_keyword_json_fn)
        for idx, row in self.df.iterrows():
            artist_name = row[self.column_name.artist]
            if artist_name in artist_list:
                self.df.loc[idx, self.column_name.is_case_study] = True
            elif any(keyword in str(row) for keyword in keyword_list):
                self.df.loc[idx, self.column_name.is_case_study] = True
            else:
                self.df.loc[idx, self.column_name.is_case_study] = False

    def _save_csv(self):
        self.df.to_csv(self.song_list_csv_fn, index=False)


@hydra.main(config_path="config", config_name="packed")
def main(config):
    annotator = LabelAnnotator(config)
    annotator.run()

if __name__ == "__main__":
    main()