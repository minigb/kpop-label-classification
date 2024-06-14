import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from pathlib import Path
import random

from utils.json import load_json
from utils.song_id import decode_song_id

class KpopDataset:
    def __init__(self, config, mode):
        self.mode = mode

        self.audio_dir = Path(config.data.audio_dir)
        self.pt_dir = Path(config.pt_dir)
        self.song_list_csv_path = Path(config.kpop_dataset.song_list_csv_fn)
        self.song_ids_json_path = Path(config.kpop_dataset.song_usage_json_fn)
        
        self.sr = config.model.cfg.sr
        self.clip_len = config.model.cfg.clip_len

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

        # # Testing with random data
        # audios = [audio for audio, _ in load_result]
        # labels = [label for _, label in load_result]
        # random.shuffle(labels)
        # self.data = list(zip(audios, labels))

    def _load_and_save_audio_segment_pt_files(self, song_id):
        def _get_n_segments():
            if self.mode == self.dict_key.valid:
                return 1
            elif self.mode == self.dict_key.test:
                _, audio_len = self._load_audio(song_id)
                return audio_len // (self.sr * self.clip_len)
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
        pt_list = self._clip_audio(audio, n_segments)
        for i, pt_path in enumerate(pt_path_list):
            pt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pt_list[i], pt_path)

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
        audio = audio.to(torch.float16)
        return audio, audio_len
    
    def _clip_audio(self, audio, n_segments):
        audio_len = audio.shape[-1]
        sample_clip_len = self.sr * self.clip_len
        ended = 0
        
        audio_segments = []
        for i in range(n_segments):
            min_start = ended
            max_start = audio_len - sample_clip_len * (n_segments - i)
            start = random.randint(min_start, max_start)
            ended = start + sample_clip_len
            assert ended <= audio_len

            audio = audio.unsqueeze(0) if len(audio.shape) == 1 else audio
            audio_segment = audio[:, start:ended]
            audio_segments.append(audio_segment)

        return audio_segments
    
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


class KpopTrainDataset(KpopDataset):
    def __init__(self, config, mode):
        super().__init__(config, mode)

    def _load_and_save_audio_segment_pt_files(self, song_id):
        pt_path = self.pt_dir / Path(f'{self.mode}/{self.n_in_channel}_{self.sr}/{song_id}.pt')
        if pt_path.exists():
            return torch.load(pt_path).to(torch.float16)
        
        audio, _ = self._load_audio(song_id)
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(audio, pt_path)
        return audio

    def __getitem__(self, idx):
        audio_whole, company_label_index, year_index = super().__getitem__(idx)
        audio_segment = self._clip_audio(audio_whole, 1)
        audio_segment = audio_segment[0]
        
        return audio_segment, company_label_index, year_index


class InferenceDataset(KpopDataset):
    def __init__(self, config, mode):
        super().__init__(config, mode)

    def __getitem__(self, idx):
        audio_segment, _ = self.data[idx]
        return audio_segment, idx

    def decode_company_label(self, index):
        return index

    def decode_year(self, index):
        return index