import torch
import torch.nn as nn
import torch.nn.functional as F

from module import SpecModel


class AudioModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.spec_converter = SpecModel(cfg.sr, cfg.n_fft, cfg.hop_length, cfg.n_mels) # shape: (batch_size, channels, n_mels, time_frames)

        self.n_in_channel = cfg.n_in_channel
        self.n_label_class = cfg.n_label_class
        self.n_year_class = cfg.n_year_class

        self.conv_layer = nn.Sequential(
            nn.Conv2d(self.n_in_channel, cfg.n_out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg.n_out_channel),
            nn.ReLU(),
            nn.MaxPool2d(3),

            nn.Conv2d(cfg.n_out_channel, cfg.n_out_channel, kernel_size=3, padding=1), 
            nn.BatchNorm2d(cfg.n_out_channel),
            nn.ReLU(),
            nn.MaxPool2d(3), 

            nn.Conv2d(cfg.n_out_channel, cfg.n_out_channel, kernel_size=3, padding=1), 
            nn.BatchNorm2d(cfg.n_out_channel),
            nn.ReLU(),
            nn.MaxPool2d(3), 
        )

        # Placeholder for fully connected layers
        self.fc_label = None
        self.fc_year = None

    def forward(self, x):
        x = self.spec_converter(x)
        x = self.conv_layer(x)
        
        # Calculate the size dynamically
        x_flat = x.view(x.size(0), -1)
        
        if self.fc_label is None:
            in_features = x_flat.size(1)
            self.fc_label = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Linear(256, self.n_label_class)
            )
            self.fc_year = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Linear(256, self.n_year_class)
            )

        label_output = self.fc_label(x_flat)
        year_output = self.fc_year(x_flat)

        return label_output, year_output