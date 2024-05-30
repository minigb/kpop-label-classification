import pandas as pd
from pathlib import Path

RELEASE_DIR = Path('releases')
ARTIST_DIR = Path('artists')
ARTIST_DIR.mkdir(exist_ok=True)

def filter_artist_from_release(label_name):
    fn = RELEASE_DIR / Path(f"{label_name.replace('/', '_')}.csv")
    assert fn.exists(), f"label_name: {label_name}. {fn} does not exist"
    df = pd.read_csv(fn)
    artists = df['artists'].unique()
    artists = [artist for artist in artists if artist != 'Various Artists']

    artists_df = pd.DataFrame({'artists': artists})
    artists_df.to_csv((ARTIST_DIR / label_name).with_suffix('.csv'), index=False)


def main():
    releases_files = list(RELEASE_DIR.glob('*.csv'))
    song_list = pd.read_csv(Path('kpop-dataset/song_list.csv'))
    labels = song_list['Label'].unique()

    for label in labels:
        label_name = label.replace('/', '_')
        if any(label_name in file.name for file in releases_files):
            filter_artist_from_release(label_name)
        else:        
            label_df = song_list[song_list['Label'] == label]
            artists = label_df['Artist'].unique()
            artists_df = pd.DataFrame({'artists': artists})
            artists_df.to_csv(ARTIST_DIR / Path(label_name + '.csv'), index=False)


def check_number():
    releases_files = list(ARTIST_DIR.glob('*.csv'))
    song_list = pd.read_csv(Path('kpop-dataset/song_list.csv'))
    labels = song_list['Label'].unique()

    print(len(releases_files), len(labels))

    # for label in labels:
    #     label_name = label.replace('/', '_')
    #     if any(label_name in file.name for file in releases_files):
    #         fn = RELEASE_DIR / Path(f"{label_name}.csv")
    #         df = pd.read_csv(fn)
    #         print(f"{label_name}: {len(df)}")
    #     else:        
    #         label_df = song_list[song_list['Label'] == label]
    #         print(f"{label_name}: {len(label_df)}")


if __name__ == '__main__':
    main()
    check_number()