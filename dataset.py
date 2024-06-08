import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from pathlib import Path
import random
import json
import pandas as pd

from utils.json import load_json
from utils.song_id import decode_song_id

class KpopDataset: # train, valid, test
    def __init__(self, config, mode):
        self.mode = mode

        self.audio_dir = Path(config.file.data.audio_dir)
        self.pt_dir = Path(config.file.pt_dir)
        self.song_list_csv_path = Path(config.file.kpop_dataset.song_list_csv_fn)
        self.song_ids_json_path = Path(config.file.kpop_dataset.song_usage_json_fn)
        
        self.sr = config.model.sr
        self.clip_len = config.model.clip_len
        self.n_clip_segment = config.model.n_clip_segment

        self.n_in_channel = config.model.n_in_channel
        self.n_label_class = config.model.n_label_class
        self.n_year_class = config.model.n_year_class

        self.dict_key = config.dict_key

        self.seed = config.data_setting.seed
        random.seed(self.seed)

        self.load_data()

    def load_data(self):
        song_usage_dict = load_json(self.song_ids_json_path)

        song_ids = []
        class_labels = [] # (company_label, year)
        for company_label, song_list in song_usage_dict[self.mode].items():
            for song_id in song_list:
                year, _, _ = decode_song_id(song_id)

                song_ids.append(song_id)
                class_labels.append((company_label, year))

        load_result = []
        for song_id, class_label in zip(song_ids, class_labels):
            audio_segments_pt = self._load_and_save_audio_segment_pt_files(song_id)
            assert len(audio_segments_pt) >= self.n_clip_segment # greater than or equal to
            load_result.extend([(audio_segment, class_label) for audio_segment in audio_segments_pt])

        self.data = load_result

    def _load_and_save_audio_segment_pt_files(self, song_id):
        n_segments = self.n_clip_segment

        pt_path_list = [self.pt_dir / Path(f'{self.mode}/{self.n_in_channel}_{self.sr}/{song_id}/{segment_num}.pt') \
                        for segment_num in range(n_segments)]
        if all([pt_path.exists() for pt_path in pt_path_list]):
            pt_list = [torch.load(pt_path) for pt_path in pt_path_list]
            return pt_list

        audio_fn = Path(f'{song_id}.mp3')
        audio, org_sr = torchaudio.load(audio_fn)
        audio_len = audio.shape[-1]
        assert audio_len >= self.sr * self.clip_len * n_segments, f'audio_len: {audio_len}, n_segments: {n_segments}'

        if org_sr != self.sr:
            audio = torchaudio.functional.resample(audio, orig_freq=org_sr, new_freq=self.sr)
        if self.n_in_channel == 1:
            audio = audio.mean(dim=0).unsqueeze(0)
        if self.mode == self.dict_key.test:
            new_n_segments = audio_len // (self.sr * self.clip_len)
            n_segments = new_n_segments
        
        sample_clip_len = self.sr * self.clip_len
        ended = 0
        pt_list = []
        for i, pt_path in enumerate(pt_path_list):
            max_start = ended + (audio.shape[-1] - sample_clip_len * (n_segments - i))
            start = random.randint(ended, max_start-1)
            ended = start + sample_clip_len

            audio = audio[:, start:ended]
            audio = audio.to(dtype=torch.float16)

            pt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(audio, pt_path)  
            pt_list.append(audio)

        return pt_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
