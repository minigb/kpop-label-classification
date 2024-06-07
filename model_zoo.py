from sympy import quadratic_congruence
import torch
import torch.nn as nn
import torchaudio

from module import SpecModel

class AudioModel(nn.Module):
  def __init__(self, sr, n_fft, hop_length, n_mels, n_in_channel, n_out_channel, n_label_class, n_year_class):
    super().__init__()
    self.sr = sr
    self.n_fft = n_fft
    self.hop_length = hop_length
    self.n_mels = n_mels
    self.spec_converter = self.SpecModel(sr, n_fft, hop_length, n_mels) # shape: (batch_size, channels, n_mels, time_frames)

    self.n_in_channel = n_in_channel
    self.n_label_class = n_label_class
    self.n_year_class = n_year_class

    self.batchnorm1d = nn.BatchNorm1d(self.n_channels*n_mels)

    # model architecture
    self.conv_layer = nn.Sequential(
        nn.Conv2d(self.n_in_channel, n_out_channel, kernel_size=3, padding=1),
        nn.BatchNorm2d(n_out_channel),
        nn.ReLU(),
        nn.MaxPool2d(3),

        nn.Conv2d(n_out_channel, n_out_channel, kernel_size=3, padding=1), 
        nn.BatchNorm2d(n_out_channel),
        nn.ReLU(),
        nn.MaxPool2d(3), 

        nn.Conv2d(n_out_channel, n_out_channel, kernel_size=3, padding=1), 
        nn.BatchNorm2d(n_out_channel),
        nn.ReLU(),
        nn.MaxPool2d(3), 
      )
    
    self.classification_head = ClassificationHead(n_out_channel, n_class, self.n_channels)

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