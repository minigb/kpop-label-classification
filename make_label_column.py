import pandas as pd
import hydra
from pathlib import Path
from tqdm import tqdm

class ArtistAndLabelMatcher:
    # Column names
    # TODO(minigb): 1) Change column names to singular form 2) control this by config
    # Need to also fix the other codes.

    ARTIST = 'artists'
    LABEL = 'labels'
    def __init__(self, config):
        self.artists_per_label_dir = Path(config.data.artists_dir)
        self.save_fn = Path(config.kpop_dataset.artist_list_csv_fn)

    def run(self):
        artists_label = self._get_artist_labels()
        self._save_result(artists_label)
    
    def _get_artist_labels(self):
        artists_label = []
        for csv_file in tqdm(list(self.artists_per_label_dir.glob('*.csv'))):
            label_name = csv_file.stem
            df = pd.read_csv(csv_file)
            for artist in df[self.ARTIST]:
                artists_label.append({self.ARTIST: artist, self.LABEL: label_name})

        return artists_label

    def _save_result(self, artists_label):
        self.save_fn.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(artists_label, columns=[self.ARTIST, self.LABEL, 'start_year', 'end_year'])
        df.to_csv(self.save_fn, index=False)


@hydra.main(config_path='config', config_name='packed')
def main(config):
    matcher = ArtistAndLabelMatcher(config)
    matcher.run()

if __name__ == '__main__':
    main()