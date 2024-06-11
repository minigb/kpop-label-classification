import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm
from pathlib import Path
import random

from utils.json import load_json
from utils.song_id import decode_song_id

class KpopDataset: # train, valid, test
    def __init__(self, config, mode):
        self.mode = mode

        self.audio_dir = Path(config.data.audio_dir)
        self.pt_dir = Path(config.pt_dir)
        self.song_list_csv_path = Path(config.kpop_dataset.song_list_csv_fn)
        self.song_ids_json_path = Path(config.kpop_dataset.song_usage_json_fn)
        
        self.sr = config.model.cfg.sr
        self.clip_len = config.model.cfg.clip_len
        self.n_clip_segment = config.model.cfg.n_clip_segment

        self.n_in_channel = config.model.cfg.n_in_channel
        self.n_label_class = config.model.cfg.n_label_class
        self.n_year_class = config.model.cfg.n_year_class

        self.dict_key = config.dict_key

        self.seed = config.data_setting.seed
        random.seed(self.seed)

        # Create mappings for labels and years
        self.company_label_to_index = {'SM': 0, 'YG': 1, 'JYP': 2, 'HYBE': 3} # TODO(minigb): Remove hardcoding
        self.index_to_company_label = {v: k for k, v in self.company_label_to_index.items()}

        # TODO(minigb): Remove hardcodings ㅠㅠ
        start_year = 1994 # TODO(minigb): Remove hardcoding
        end_year = 2024 # TODO(minigb): Remove hardcoding
        if self.n_year_class != 31:
            start_year += 1
        years_per_class = (end_year + 1 - start_year) // self.n_year_class
        self.year_to_index = {1994 : 0} # TODO(minigb): Remove hardcoding
        for year in range(start_year, end_year + 1):
            self.year_to_index[year] = (year - start_year) // years_per_class
        self.index_to_year = {v: start_year + v * years_per_class for v in range(self.n_year_class)}

        self.load_data()

    def load_data(self):
        song_usage_dict = load_json(self.song_ids_json_path)

        song_ids = []
        class_labels = [] # (company_label, year)
        for company_label, song_list in song_usage_dict[self.mode].items():
            for song_id in song_list:
                year, _, _ = decode_song_id(song_id)

                song_ids.append(song_id)
                class_labels.append((company_label, int(year)))

        load_result = []
        assert len(song_ids) == len(class_labels)
        for song_id, class_label in tqdm(zip(song_ids, class_labels), total=len(song_ids), desc=f'Loading {self.mode} data'):
            audio_segments_pt = self._load_and_save_audio_segment_pt_files(song_id)
            assert len(audio_segments_pt) >= 1 # greater than or equal to
            load_result.extend([(audio_segment, class_label) for audio_segment in audio_segments_pt])

        self.data = load_result

    def _load_and_save_audio_segment_pt_files(self, song_id):
        def _get_n_segments():
            if self.mode == self.dict_key.train:
                return self.n_clip_segment
            elif self.mode == self.dict_key.valid:
                return 1
            elif self.mode == self.dict_key.test:
                _, audio_len = self._load_audio(song_id)
                return audio_len // (self.sr * self.clip_len)
                # return 1
            else:
                raise ValueError(f'Invalid mode: {self.mode}')
        
        # return pt files if already exists
        n_segments = _get_n_segments()
        pt_path_list = [self.pt_dir / Path(f'{self.mode}/{self.n_in_channel}_{self.sr}/{self.clip_len}s_{n_segments}segs/{song_id}/{segment_num}.pt') \
                        for segment_num in range(n_segments)]
        if all([pt_path.exists() for pt_path in pt_path_list]):
            return [torch.load(pt_path).to(torch.float16) for pt_path in pt_path_list]
        
        # load audio
        audio, audio_len = self._load_audio(song_id)
        assert audio_len >= self.sr * self.clip_len * n_segments, f'audio_len: {audio_len}, n_segments: {n_segments}'

        # clipping
        sample_clip_len = self.sr * self.clip_len
        ended = 0
        pt_list = []
        for i, pt_path in enumerate(pt_path_list):
            min_start = ended
            max_start = audio_len - sample_clip_len * (n_segments - i)
            start = random.randint(min_start, max_start)
            ended = start + sample_clip_len
            assert ended <= audio_len

            audio_segment = audio[:, start:ended]
            audio_segment = audio_segment.to(dtype=torch.float16)
            assert audio_segment.shape == (self.n_in_channel, sample_clip_len)

            pt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(audio_segment, pt_path)
            pt_list.append(audio_segment)

        return pt_list

    def _load_audio(self, song_id):
        audio_fn = Path(f'{self.audio_dir}/{song_id}.mp3')
        assert audio_fn.exists(), f'{audio_fn} does not exist'
        audio, org_sr = torchaudio.load(audio_fn)
        if org_sr != self.sr:
            audio = torchaudio.functional.resample(audio, orig_freq=org_sr, new_freq=self.sr)
        if self.n_in_channel == 1:
            audio = audio.mean(dim=0).unsqueeze(0)
        audio_len = audio.shape[-1]
        return audio, audio_len
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_segment, (company_label, year) = self.data[idx]
        company_label_index = self.company_label_to_index[company_label]
        year_index = self.year_to_index[year]
        return audio_segment, company_label_index, year_index

    def decode_company_label(self, index):
        return self.index_to_company_label[index]

    def decode_year(self, index):
        return self.index_to_year[index]
