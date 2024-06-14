import torch
import torch.nn as nn
import hydra

from module import SpecModel


class Basic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_in_channel = cfg.n_in_channel
        self.n_out_channel = cfg.n_out_channel
        self.n_conv_layer = cfg.n_conv_layer

        self.n_label_class = cfg.n_label_class
        self.n_year_class = cfg.n_year_class

        self.spec_converter = SpecModel(cfg.sr, cfg.n_fft, cfg.hop_length, cfg.n_mels)  # shape: (batch_size, channels, n_mels, time_frames)

        self.label_dim_proportion = cfg.label_dim_proportion if hasattr(cfg, 'label_dim_proportion') else 0.5

        self._init_conv_layers(cfg)
        self._init_fc_layers(cfg)

    def _init_conv_layers(self, cfg):
        layers = []
        in_channel = self.n_in_channel
        out_channel = self.n_out_channel
        for _ in range(self.n_conv_layer):
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=cfg.kernel_size, padding=1))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(cfg.max_pool_size))
            in_channel = out_channel  # Update in_channel for the next layer
        self.conv_layer = nn.Sequential(*layers)
    
    def _init_fc_layers(self, cfg):
        def _get_final_dim(n_mels, max_pool_size, n_conv_layer):
            final_dim = n_mels
            for _ in range(n_conv_layer):
                final_dim = final_dim // max_pool_size
            return self.n_out_channel * final_dim

        final_dim = _get_final_dim(cfg.n_mels, cfg.max_pool_size, self.n_conv_layer)
        fc_label_dim = int(final_dim * self.label_dim_proportion)
        fc_year_dim = final_dim - fc_label_dim

        if fc_label_dim and self.n_label_class:
            self.fc_label = nn.Linear(fc_label_dim, self.n_label_class)
        if fc_year_dim and self.n_year_class:
            self.fc_year = nn.Linear(final_dim - fc_label_dim, self.n_year_class)

    def forward(self, x):
        spec = self.spec_converter(x)  # (batch_size, channels, n_mels, time_frames)

        out = self.conv_layer(spec)  # (batch_size, n_out_channel, n_mels // 27, time_frames // 27)
        out = out.flatten(1, 2)  # (batch_size, n_out_channel * n_mels // 27, time_frames // 27)
        out = out.mean(dim=-1)  # (batch_size, n_out_channel * n_mels // 27)

        label_dim = int(out.size(1) * self.label_dim_proportion)
        label_output = self.fc_label(out[:, :label_dim]) if hasattr(self, 'fc_label') else None
        year_output = self.fc_year(out[:, label_dim:]) if hasattr(self, 'fc_year') else None

        return label_output, year_output

# class YearOnly(Basic):
#     def __init__(self, cfg):
#         super().__init__(cfg)
 
#     def _init_fc_layers(self, cfg):
#         def _get_final_dim(n_mels, max_pool_size, n_conv_layer):
#             final_dim = n_mels
#             for _ in range(n_conv_layer):
#                 final_dim = final_dim // max_pool_size
#             return self.n_out_channel * final_dim

#         final_dim = _get_final_dim(cfg.n_mels, cfg.max_pool_size, self.n_conv_layer)
#         fc_label_dim = 0
#         self.fc_label = nn.Linear(fc_label_dim, self.n_label_class)
#         self.fc_year = nn.Linear(final_dim - fc_label_dim, self.n_year_class)

#     def forward(self, x):
#         spec = self.spec_converter(x)  # (batch_size, channels, n_mels, time_frames)

#         out = self.conv_layer(spec)  # (batch_size, n_out_channel, n_mels // 27, time_frames // 27)
#         out = out.flatten(1, 2)  # (batch_size, n_out_channel * n_mels // 27, time_frames // 27)
#         out = out.mean(dim=-1)  # (batch_size, n_out_channel * n_mels // 27)

#         label_output = self.fc_label(out[:, :out.size(1) // 2])
#         year_output = self.fc_year(out[:, out.size(1) // 2:])

#         return label_output, year_output
