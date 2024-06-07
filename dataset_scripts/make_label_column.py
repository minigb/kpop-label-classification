import pandas as pd
import hydra
from pathlib import Path
from tqdm import tqdm
from collections import Counter

class ArtistAndLabelMatcher:
    def __init__(self, config):
        self.artists_per_label_dir = Path(config.data.artists_dir)
        self.save_fn = Path(config.kpop_dataset.artist_list_csv_fn)
        self.column_name = config.column_name

    def run(self):
        artist_label_list = self._get_artist_labels()
        artist_label_df = self._get_df_with_year_added(artist_label_list)
        result_df = self._merge_original_and_updated_df(artist_label_df)
        self._save_result(result_df)
    
    def _get_artist_labels(self):
        artist_label = []
        for csv_file in tqdm(list(self.artists_per_label_dir.glob('*.csv'))):
            label_name = csv_file.stem
            df = pd.read_csv(csv_file)
            for artist in df[self.column_name.artist]:
                artist_label.append({self.column_name.artist: artist, self.column_name.label: label_name})

        return artist_label

    def _get_df_with_year_added(self, artist_label_list):
        counter = Counter([x[self.column_name.artist] for x in artist_label_list])
        for artist_and_label in artist_label_list:
            artist_name = artist_and_label[self.column_name.artist]
            artist_and_label[self.column_name.label_start] = artist_and_label[self.column_name.label_end] = 0 if counter[artist_name] > 1 else None

        df = pd.DataFrame(artist_label_list)
        df[self.column_name.label_start] = df[self.column_name.label_start].astype("Int32")
        df[self.column_name.label_end] = df[self.column_name.label_end].astype("Int32")
        
        df = df.sort_values(by=[self.column_name.artist, self.column_name.label])
        return df

    def _merge_original_and_updated_df(self, new_df):
        if not self.save_fn.exists():
            return new_df
        original_df = pd.read_csv(self.save_fn)

        # original_df: if artist and label does not exist in the new_df, remove it.
        idx_to_remove = []
        for idx, row in original_df.iterrows():
            # if empty
            if new_df[(new_df[self.column_name.artist] == row[self.column_name.artist]) & (new_df[self.column_name.label] == row[self.column_name.label])].empty:
                idx_to_remove.append(idx)
        original_df = original_df.drop(idx_to_remove)

        # new_df: if artist and label already exist in the original_df, remove it.
        idx_to_remove = []
        for idx, row in new_df.iterrows():
            # if not empty
            if not original_df[(original_df[self.column_name.artist] == row[self.column_name.artist]) & (original_df[self.column_name.label] == row[self.column_name.label])].empty:
                idx_to_remove.append(idx)
        new_df = new_df.drop(idx_to_remove)

        merged_df = pd.concat([original_df, new_df])

        return merged_df

    def _save_result(self, df):
        self.save_fn.parent.mkdir(parents=True, exist_ok=True)
        
        df = df.sort_values(by=[self.column_name.artist, self.column_name.label])
        df.to_csv(self.save_fn, index=False)


@hydra.main(config_path='config', config_name='packed')
def main(config):
    matcher = ArtistAndLabelMatcher(config)
    matcher.run()

if __name__ == '__main__':
    main()