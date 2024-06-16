import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from pathlib import Path
import random

from utils.json import load_json
from utils.song_id import decode_song_id

class KpopDataset:
    def __init__(self, config, mode, audio_type = 'mp3'):
        self.mode = mode
        self.audio_type = audio_type

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
        audio_fn = Path(f'{self.audio_dir}/{song_id}.{self.audio_type}')
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
    

class GTZANDataset:
  # https://github.com/jdasam/ant5015/blob/main/notebooks/4th_week_genre_classification.ipynb
  def __init__(self, config, mode):
    self.data_dir = Path('genres') # TODO(minigb): Remove hardcoding
    self.pt_dir = Path('pt_gtzan') # TODO(minigb): Remove hardcoding
    self.is_test = (not mode == 'train')

    self.wav_fns = list(self.data_dir.rglob('*.wav'))
    self.wav_fns.sort() # sort before shuffling
    random.seed(0) # fix random seed
    random.shuffle(self.wav_fns)# shuffle list

    assert len(self.wav_fns) > 100

    self.orig_freq = 22050
    self.target_sr = config.model.cfg.sr
    self.resampler = torchaudio.transforms.Resample(orig_freq=self.orig_freq, new_freq=self.target_sr)

    self.audio_label_pairs = self.load_audio()
    self.class_names = self.make_class_vocab()
    self.slice_dur = config.model.cfg.clip_len
    self.slice_samples = self.slice_dur * self.target_sr

    # self.labels = self.load_label()

  def load_audio(self):
    audios = []
    selected_fns = self.wav_fns[800:] if self.is_test else self.wav_fns[:800]
    for wav_fn in tqdm(selected_fns):
      pt_path = self.pt_dir / Path(f'{wav_fn.stem}.pt')
      if pt_path.exists():
        y = torch.load(pt_path)
      else:
          y, sr = torchaudio.load(wav_fn)
          assert sr == self.orig_freq, f'SR has to besame with expected frequency {self.orig_freq}'
          y = self.resampler(y)
          y = y.to(torch.float16)
          pt_path.parent.mkdir(parents=True, exist_ok=True)
          torch.save(y, pt_path)
      audios.append([y, wav_fn.parent.name])
    return audios

  def make_class_vocab(self):
    class_names = [l for a, l in self.audio_label_pairs]
    # class_names = [pair[1] for pair in self.audio_label_pairs]
    class_names = sorted(list(set(class_names)))
    return class_names

  # Dataset class에 꼭 필요한 두가지: __len__, __getitem__
  def __len__(self):
    return len(self.audio_label_pairs)

  def __getitem__(self, idx:int):
    # dataset[idx] 을 불렀을 때 호출되는 함수
    # idx-th 아이템을 반환해주는 함수
    audio, label = self.audio_label_pairs[idx]

    # string이 아닌 텐서로 반환해줄 수 있도록 변환
    num_samples = audio.shape[-1]
    slice_end = num_samples - self.slice_samples
    slice_start = random.randint(0, slice_end-1) #randint는 끝점 포함


    return audio[:, slice_start:slice_start+self.slice_samples], 0, self.class_names.index(label) # returning dummy 0
