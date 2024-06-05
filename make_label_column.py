import pandas as pd
import hydra
from pathlib import Path
from tqdm import tqdm
from collections import Counter

class ArtistAndLabelMatcher:
    # Column names
    # TODO(minigb): 1) Change column names to singular form 2) control this by config
    # Need to also fix the other codes.

    ARTIST = 'artists'
    LABEL = 'labels'
    START = 'start_date'
    END = 'end_date'
    def __init__(self, config):
        self.artists_per_label_dir = Path(config.data.artists_dir)
        self.save_fn = Path(config.kpop_dataset.artist_list_csv_fn)

    def run(self):
        artist_label_list = self._get_artist_labels()
        artist_label_df = self._get_df_with_year_added(artist_label_list)
        self._save_result(artist_label_df)
    
    def _get_artist_labels(self):
        artist_label = []
        for csv_file in tqdm(list(self.artists_per_label_dir.glob('*.csv'))):
            label_name = csv_file.stem
            df = pd.read_csv(csv_file)
            for artist in df[self.ARTIST]:
                artist_label.append({self.ARTIST: artist, self.LABEL: label_name})

        return artist_label

    def _get_df_with_year_added(self, artist_label_list):
        counter = Counter([x[self.ARTIST] for x in artist_label_list])
        for artist_and_label in artist_label_list:
            artist_name = artist_and_label[self.ARTIST]
            artist_and_label[self.START] = artist_and_label[self.END] = 0 if counter[artist_name] > 1 else None

        df = pd.DataFrame(artist_label_list)
        df[self.START] = df[self.START].astype("Int32")
        df[self.END] = df[self.END].astype("Int32")
        
        df = df.sort_values(by=[self.ARTIST, self.LABEL])
        return df

    def _save_result(self, df):
        self.save_fn.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.save_fn, index=False)


@hydra.main(config_path='config', config_name='packed')
def main(config):
    matcher = ArtistAndLabelMatcher(config)
    matcher.run()

if __name__ == '__main__':
    main()