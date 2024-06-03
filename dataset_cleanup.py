import pandas as pd
from fuzzywuzzy import fuzz
from pathlib import Path
from tqdm import tqdm
from utils.song_id import get_song_id

CSV_DIR = Path('csv')
FUZZ_THRESHOLD = 70
ALMOST_EQUAL_THRESHOLD = 95
TOKEN_SORT_THRESHOLD = 95


class CsvColumnNames:
  def __init__(self, date, title, artist):
    self.date = date
    self.title = title
    self.artist = artist

column_names = CsvColumnNames('Year', 'Song', 'Artist')

class Info:
    def __init__(self, row):
        self.song_title = row[column_names.title].lower()
        self.artist = row[column_names.artist].lower()
        self.original_video_title = row['video_title'].lower()
        self.original_video_channel = row['video_channel'].lower()

        self.is_topic = row['topic']
        self.is_artist_name_in_title = True if self.artist in self.original_video_title else False

        # refinement
        self.video_title = self._refine_video_title(self.original_video_title)
        self.video_channel = self._refine_video_channel(self.original_video_channel)

        self.song_fuzz_ratio = fuzz.ratio(self.song_title, self.video_title)
        self.artist_fuzz_ratio = max(fuzz.ratio(self.artist, self.video_channel),fuzz.ratio(self.artist, self.video_title))
        self.song_partial_ratio = fuzz.partial_ratio(self.song_title, self.video_title)
        self.artist_partial_ratio = max(fuzz.partial_ratio(self.artist, self.video_channel),fuzz.partial_ratio(self.artist, self.video_title))
        self.song_token_ratio = fuzz.token_set_ratio(self.song_title, self.video_title)
        

    def _refine_video_title(self, video_title):
        words_to_remove = [
            'official audio',
            self.artist,
            # 'official video',
            # 'official music video',
            'audio',
            # 'video',
            # 'music video',
            '()',
            '[]',
        ]
        refined = video_title.lower()
        for word in words_to_remove:
            refined = refined.replace(word, '')
        return refined
    
    def _refine_video_channel(self, video_channel):
        refined = video_channel.lower().replace(' - topic', '')
        return refined
    
class DatasetCleaner:
    def __init__(self):
        return
    
    # exact match
    def _song_almost_exact_match(self, row, threshold = ALMOST_EQUAL_THRESHOLD):
        info = Info(row)
        return info.song_partial_ratio >= threshold
    
    def _artist_almost_exact_match(self, row, threshold = ALMOST_EQUAL_THRESHOLD):
        info = Info(row)
        return info.artist_partial_ratio >= threshold
    
    # fuzz above threshold
    def _song_fuzz_above_threshold(self, row, threshold = FUZZ_THRESHOLD):
        info = Info(row)
        return info.song_fuzz_ratio >= threshold
    
    def _artist_fuzz_above_threshold(self, row, threshold = FUZZ_THRESHOLD):
        info = Info(row)
        return info.artist_fuzz_ratio >= threshold
    
    # token set ratio
    def _song_token_ratio_above_threshold(self, row, threshold = TOKEN_SORT_THRESHOLD):
        info = Info(row)
        return info.song_token_ratio >= threshold


    # wrapper
    def is_clean(self, row):
        is_clean = False
        is_clean = is_clean or (self._song_almost_exact_match(row) or self._song_fuzz_above_threshold(row) or self._song_token_ratio_above_threshold(row)) \
            and (self._artist_almost_exact_match(row) or self._artist_fuzz_above_threshold(row) or row['topic'])
        is_clean = is_clean or (self._song_almost_exact_match(row))
        return is_clean


def get_rows_to_remove(original_csv_path, things_to_remove_csv_path):
    df = pd.read_csv(original_csv_path)
    cleaner = DatasetCleaner()

    idx_to_keep = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if cleaner.is_clean(row):
            idx_to_keep.append(idx)
    
    df_to_remove = df.drop(idx_to_keep)

    # Add information and sort
    df_to_remove
    for idx, row in df_to_remove.iterrows():
        info = Info(row)
        df_to_remove.loc[idx, 'song_fuzz'] = info.song_fuzz_ratio
        df_to_remove.loc[idx, 'song_partial_fuzz'] = info.song_partial_ratio
    df_to_remove.to_csv(things_to_remove_csv_path, index=False)
    

def move_audio_files(from_dir, to_dir, original_csv_path, things_to_remove_csv_path):
    to_dir.mkdir(parents=True, exist_ok=True)

    original_df = pd.read_csv(original_csv_path)
    remove_df = pd.read_csv(things_to_remove_csv_path)
    for _, row in tqdm(remove_df.iterrows(), total=len(remove_df)):
        date = row[column_names.date]
        song = row[column_names.title]
        artist = row[column_names.artist]
        
        song_id = get_song_id(date, song, artist)
        audio_path = from_dir / Path(f'{song_id}.mp3')
        if audio_path.exists():
            audio_path.rename(to_dir / Path(f'{song_id}.mp3'))

        original_df.drop(original_df[(original_df[column_names.date] == date) \
                                     & (original_df[column_names.title] == song) \
                                        & (original_df[column_names.artist] == artist)].index, inplace=True)
    original_df.to_csv(original_csv_path, index=False)

    files_in_from_dir = list(from_dir.glob('*.mp3'))
    assert len(original_df) == len(files_in_from_dir), f"original_df has {len(original_df)} rows, but there are {len(files_in_from_dir)} audio files"
    files_in_to_dir = list(to_dir.glob('*.mp3'))
    assert len(remove_df) == len(files_in_to_dir), f"remove_df has {len(remove_df)} rows, but there are {len(files_in_to_dir)} audio files"


if __name__ == '__main__':
    from_dir = Path('audio')
    to_dir = Path('audio_tmp')
    original_csv_path = CSV_DIR / Path('kpop_chosen.csv')
    things_to_remove_csv_path = CSV_DIR / Path('kpop_things_to_remove.csv')

    get_rows_to_remove(original_csv_path, things_to_remove_csv_path)
    move_audio_files(from_dir, to_dir, original_csv_path, things_to_remove_csv_path)