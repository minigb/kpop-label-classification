import torch.nn as nn
import torchaudio

class SpecModel(nn.Module):
    def __init__(self, sr:int, n_fft:int, hop_length:int, n_mels:int):
        super().__init__()
        self.mel_converter = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        self.db_converter = torchaudio.transforms.AmplitudeToDB()

    def forward(self, x):
        mel_spec = self.mel_converter(x)
        return self.db_converter(mel_spec)