import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import hydra
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random

from utils.song_id import get_song_id
from utils.json import load_json, save_json

class Categorizer:
    def __init__(self, config):
        self.song_list_csv_fn = Path(config.kpop_dataset.song_list_csv_fn)
        self.song_usage_json_fn = Path(config.kpop_dataset.song_usage_json_fn)
        self.audio_dir = Path(config.data.audio_dir)

        self.column_name = config.column_name
        self.dict_key = config.dict_key

        self.train_config = config.train

        # case study
        self.case_study_fn = Path(config.kpop_dataset.type.case_study_fn)

        # major label
        self.major_label_json_fn = Path(config.kpop_dataset.type.major_label_fn)

        # init
        self.df = pd.read_csv(self.song_list_csv_fn)
        self.result_dict = {}

    def run(self):
        self._drop_songs_without_audio()
        self._check_case_study()
        self._check_major_label()
        self._split_train_test()
        self._select_inference_songs()
        self._save_result() # save result

    def _drop_songs_without_audio(self):
        idx_to_remove = []
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            song_id = self._get_song_id(row)
            audio_fn = self.audio_dir / f'{song_id}.mp3'
            if not audio_fn.exists():
                idx_to_remove.append(idx)
        self.df = self.df.drop(idx_to_remove)

    def _check_case_study(self):
        case_study_type_dict = load_json(self.case_study_fn)
        
        song_ids_dict = defaultdict(list)
        case_study_idx = []
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            song_id = self._get_song_id(row)

            row_str = ' '.join(map(str, row.dropna().values)).lower()
            for case_study_type, case_study_list in case_study_type_dict.items():
                for keyword in case_study_list:
                    if f' {keyword} '.lower() in row_str.lower():
                        case_study_idx.append(idx)
                        song_ids_dict[case_study_type].append(song_id)
                        break
        
        self.result_dict[self.dict_key.case_study] = song_ids_dict
        self.df = self.df.drop(case_study_idx)

    def _check_major_label(self):
        major_label_df = self.df[self.df[self.column_name.is_major_label]]
        label_dict = load_json(self.major_label_json_fn)

        song_ids_dict = defaultdict(list)
        for _, row in tqdm(major_label_df.iterrows(), total=len(major_label_df)):
            if not row[self.column_name.is_major_label]:
                continue
            
            song_id = self._get_song_id(row)
            
            for label_representation, labels_included_list in label_dict.items():
                for label_name in labels_included_list:
                    if row[self.column_name.label] == label_name:
                        song_ids_dict[label_representation].append(song_id)
                        break

        self.result_dict[self.dict_key.major_label] = song_ids_dict

    def _split_train_test(self):
        random.seed(self.train_config.seed)

        # get minimum size
        label_dict = self.result_dict[self.dict_key.major_label]
        min_size = min([len(song_ids_list) for song_ids_list in label_dict.values()])

        # split
        train_song_ids_dict = {}
        val_song_ids_dict = {}
        test_song_ids_dict = {}
        
        val_ratio = self.train_config.val_ratio
        test_ratio = self.train_config.test_ratio
        for label_representation, song_ids_list in label_dict.items():
            songs_used = random.sample(song_ids_list, min_size)

            train, test = train_test_split(songs_used, test_size=test_ratio, random_state=self.train_config.seed)
            train, valid = train_test_split(train, test_size=val_ratio/(1-test_ratio), random_state=self.train_config.seed)
            
            train_song_ids_dict[label_representation] = train
            val_song_ids_dict[label_representation] = valid
            test_song_ids_dict[label_representation] = test

            assert len(train) + len(valid) + len(test) == len(songs_used)

        self.result_dict[self.dict_key.train] = train_song_ids_dict
        self.result_dict[self.dict_key.valid] = val_song_ids_dict
        self.result_dict[self.dict_key.test] = test_song_ids_dict
        del self.result_dict[self.dict_key.major_label] # no need to save major label anymore

    def _select_inference_songs(self):
        non_major_label_df = self.df[self.df[self.column_name.is_major_label] == False]
        random_pick_song = non_major_label_df.sample(n=self.train_config.inference_size, random_state=self.train_config.seed)

        song_ids_dict = {'_': # dummy key
                         [self._get_song_id(row) for _, row in random_pick_song.iterrows()]}
        self.result_dict[self.dict_key.inference] = song_ids_dict

    def _get_song_id(self, row):
        year = row[self.column_name.song.year]
        song = row[self.column_name.song.title]
        artist = row[self.column_name.song.artist]
        return get_song_id(year, song, artist)

    def _save_result(self):
        for key, value in self.result_dict.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    value[k] = sorted(v)
            elif isinstance(value, list):
                self.result_dict[key] = sorted(value)

        self.song_usage_json_fn.parent.mkdir(parents=True, exist_ok=True)
        save_json(self.song_usage_json_fn, self.result_dict)


@hydra.main(config_path="../config", config_name="packed")
def main(config):
    categorizer = Categorizer(config)
    categorizer.run()

if __name__ == "__main__":
    main()