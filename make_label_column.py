import pandas as pd
import hydra
from pathlib import Path
from tqdm import tqdm

class ArtistAndLabelMatcher:
    def __init__(self, config):
        self.artists_per_label_dir = Path(config.data.artists_dir)
        self.save_fn = Path(config.kpop_dataset.artist_list_csv_fn)

    def run(self):
        artists_label = self._get_artist_labels()
        self._save_result(artists_label)
    
    def _get_artist_labels(self):
        artists_label = {}
        for csv_file in tqdm(list(self.artists_per_label_dir.glob('*.csv'))):
            label_name = csv_file.stem
            df = pd.read_csv(csv_file)
            for artist in df['artists']:
                if artist not in artists_label:
                    artists_label[artist] = []
                artists_label[artist].append(label_name)

        return artists_label

    def _save_result(self, artists_label):
        self.save_fn.parent.mkdir(parents=True, exist_ok=True)
        # TODO(minigb): Change column names to singular form
        # Need to also fix the other codes for this.
        df = pd.DataFrame(artists_label.items(), columns=['artists', 'labels'])
        df.to_csv(self.save_fn, index=False)


@hydra.main(config_path='config', config_name='packed')
def main(config):
    matcher = ArtistAndLabelMatcher(config)
    matcher.run()

if __name__ == '__main__':
    main()