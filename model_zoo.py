from sympy import quadratic_congruence
import torch
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


class AudioModel(nn.Module):
  def __init__(self, sr, channel, n_fft, hop_length, n_mels, out_channels, n_class):
    super().__init__()
    self.sr = sr
    self.channel = channel
    self.n_channels = {'mono': 1, 'stereo': 2}[self.channel]
    self.spec_converter = self.SpecModel(sr, n_fft, hop_length, n_mels) # shape: (batch_size, channels, n_mels, time_frames)
    self.batchnorm1d = nn.BatchNorm1d(self.n_channels*n_mels)

    # model architecture
    self.conv_layer = nn.Sequential( 
        nn.Conv2d(self.n_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(3),

        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), 
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(3), 

        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), 
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(3), 
      )
    
    self.classification_head = ClassificationHead(out_channels, n_class, self.n_channels)

  def get_spec(self, x):
    return self.spec_converter(x)
  
  def forward(self, x, return_embedding=False):
    spec = self.get_spec(x)
    spec_reshaped = spec.view(spec.shape[0], -1, spec.shape[-1])
    x = self.batchnorm1d(spec_reshaped)
    x = x.view(spec.shape[0], spec.shape[1], spec.shape[2], spec.shape[3])
    x = self.conv_layer(spec)
    x = x.flatten(1, 2)
    x = x.mean(dim=-1)
    if return_embedding:
      return x 
    return self.classification_head(x)