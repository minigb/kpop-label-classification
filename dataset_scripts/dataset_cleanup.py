import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from tqdm.auto import tqdm
from fuzzywuzzy import fuzz
from pathlib import Path
import hydra

from utils.song_id import get_song_id


class RowInfo:
    def __init__(self, column_names, row):
        self.song_title = row[column_names.title].lower()
        self.artist = row[column_names.artist].lower()
        self.original_video_title = row['video_title'].lower()
        self.original_video_channel = row['video_channel'].lower()

        self.is_topic = ' - topic' in self.original_video_channel
        self.is_artist_name_in_title = self.artist in self.original_video_title

        # refinement
        self.video_title = self._refine_video_title(self.original_video_title)
        self.video_channel = self._refine_video_channel(self.original_video_channel)

        self.song_fuzz_ratio = fuzz.ratio(self.song_title, self.video_title)
        self.artist_fuzz_ratio = max(fuzz.ratio(self.artist, self.video_channel), fuzz.ratio(self.artist, self.video_title))
        self.song_partial_ratio = fuzz.partial_ratio(self.song_title, self.video_title)
        self.artist_partial_ratio = max(fuzz.partial_ratio(self.artist, self.video_channel), fuzz.partial_ratio(self.artist, self.video_title))
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
        return video_channel.lower().replace(' - topic', '')


class AudioDownloadCleaner:
    class _DatasetCleaner:
        FUZZ_THRESHOLD = 70
        ALMOST_EQUAL_THRESHOLD = 95
        TOKEN_SORT_THRESHOLD = 95
        
        def __init__(self, column_names):
            self.column_names = column_names
            pass

        def _song_almost_exact_match(self, row, threshold=ALMOST_EQUAL_THRESHOLD):
            row_info = RowInfo(self.column_names, row)
            return row_info.song_partial_ratio >= threshold

        def _artist_almost_exact_match(self, row, threshold=ALMOST_EQUAL_THRESHOLD):
            row_info = RowInfo(self.column_names, row)
            return row_info.artist_partial_ratio >= threshold

        def _song_fuzz_above_threshold(self, row, threshold=FUZZ_THRESHOLD):
            row_info = RowInfo(self.column_names, row)
            return row_info.song_fuzz_ratio >= threshold

        def _artist_fuzz_above_threshold(self, row, threshold=FUZZ_THRESHOLD):
            row_info = RowInfo(self.column_names, row)
            return row_info.artist_fuzz_ratio >= threshold

        def _song_token_ratio_above_threshold(self, row, threshold=TOKEN_SORT_THRESHOLD):
            row_info = RowInfo(self.column_names, row)
            return row_info.song_token_ratio >= threshold
        
        def _is_topic(self, row):
            row_info = RowInfo(self.column_names, row)
            return row_info.is_topic

        def is_clean(self, row):
            is_clean = False
            is_clean = is_clean or (self._song_almost_exact_match(row) or self._song_fuzz_above_threshold(row) or self._song_token_ratio_above_threshold(row)) \
                and (self._artist_almost_exact_match(row) or self._artist_fuzz_above_threshold(row) or self._is_topic(row))
            is_clean = is_clean or (self._song_almost_exact_match(row))
            return is_clean

    def __init__(self, config):
        self.from_dir = Path(config.data.audio_dir)
        self.to_dir = Path(config.data.removed_audio_dir)
        self.original_csv_path = Path(config.kpop_dataset.save_csv_fns.chosen)
        self.things_to_remove_csv_path = Path(f'{config.kpop_dataset.save_csv_name}_to_remove.csv')
        self.column_names = config.csv_column_names.video

    def check_if_audio_files_exist(self):
        df = pd.read_csv(self.original_csv_path)
        idx_to_remove = []
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            year = row[self.column_names.year]
            song = row[self.column_names.title]
            artist = row[self.column_names.artist]

            song_id = get_song_id(year, song, artist)
            audio_path = self.from_dir / Path(f'{song_id}.mp3')
            if not audio_path.exists():
                idx_to_remove.append(idx)
        df.drop(idx_to_remove, inplace=True)
        df.to_csv(self.original_csv_path, index=False)

    def get_rows_to_remove(self):
        df = pd.read_csv(self.original_csv_path)
        cleaner = self._DatasetCleaner(self.column_names)

        idx_to_keep = []

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            if cleaner.is_clean(row):
                idx_to_keep.append(idx)

        df_to_remove = df.drop(idx_to_keep)

        for idx, row in df_to_remove.iterrows():
            row_info = RowInfo(self.column_names, row)
            df_to_remove.loc[idx, 'song_fuzz'] = row_info.song_fuzz_ratio
            df_to_remove.loc[idx, 'song_partial_fuzz'] = row_info.song_partial_ratio
        df_to_remove.to_csv(self.things_to_remove_csv_path, index=False)

    def move_audio_files(self):
        self.to_dir.mkdir(parents=True, exist_ok=True)

        original_df = pd.read_csv(self.original_csv_path)
        remove_df = pd.read_csv(self.things_to_remove_csv_path)
        for _, row in tqdm(remove_df.iterrows(), total=len(remove_df)):
            year = row[self.column_names.year]
            song = row[self.column_names.title]
            artist = row[self.column_names.artist]

            song_id = get_song_id(year, song, artist)
            audio_path = self.from_dir / Path(f'{song_id}.mp3')
            if audio_path.exists():
                audio_path.rename(self.to_dir / Path(f'{song_id}.mp3'))

            original_df.drop(original_df[(original_df[self.column_names.year] == year) \
                                         & (original_df[self.column_names.title] == song) \
                                         & (original_df[self.column_names.artist] == artist)].index, inplace=True)
        original_df.to_csv(self.original_csv_path, index=False)

        files_in_from_dir = list(self.from_dir.glob('*.mp3'))
        assert len(original_df) == len(files_in_from_dir), f"original_df has {len(original_df)} rows, but there are {len(files_in_from_dir)} audio files"
        files_in_to_dir = list(self.to_dir.glob('*.mp3'))
        assert len(remove_df) == len(files_in_to_dir), f"remove_df has {len(remove_df)} rows, but there are {len(files_in_to_dir)} audio files"

    def run(self):
        self.check_if_audio_files_exist()
        self.get_rows_to_remove()
        self.move_audio_files()


@hydra.main(config_path="../config", config_name='packed')
def main(config):
    processor = AudioDownloadCleaner(config)
    processor.run()


if __name__ == '__main__':
    main()
