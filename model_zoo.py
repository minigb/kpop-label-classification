import torch
import torch.nn as nn
import hydra

from module import SpecModel


class AudioModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_in_channel = cfg.n_in_channel
        self.n_out_channel = cfg.n_out_channel

        self.n_label_class = cfg.n_label_class
        self.n_year_class = cfg.n_year_class

        self.spec_converter = SpecModel(cfg.sr, cfg.n_fft, cfg.hop_length, cfg.n_mels)  # shape: (batch_size, channels, n_mels, time_frames)
        self.batchnorm1d = nn.BatchNorm1d(self.n_in_channel * cfg.n_mels)

        # input: (batch_size, n_in_channel, n_mels, time_frames)
        self.conv_layer = nn.Sequential(
            nn.Conv2d(self.n_in_channel, self.n_out_channel, kernel_size=cfg.kernel_size, padding=1),
            nn.BatchNorm2d(self.n_out_channel),
            nn.ReLU(),
            nn.MaxPool2d(cfg.max_pool_size), # result: (batch_size, n_out_channel, n_mels // 3, time_frames // 3)

            nn.Conv2d(self.n_out_channel, self.n_out_channel, kernel_size=cfg.kernel_size, padding=1), 
            nn.BatchNorm2d(self.n_out_channel),
            nn.ReLU(),
            nn.MaxPool2d(cfg.max_pool_size), # result: (batch_size, n_out_channel, n_mels // 9, time_frames // 9)

            nn.Conv2d(self.n_out_channel, self.n_out_channel, kernel_size=cfg.kernel_size, padding=1), 
            nn.BatchNorm2d(self.n_out_channel),
            nn.ReLU(),
            nn.MaxPool2d(cfg.max_pool_size), # result: (batch_size, n_out_channel, n_mels // 27, time_frames // 27)
        )

        # Initialize the fully connected layers
        final_dim = self.n_out_channel * (cfg.n_mels // cfg.max_pool_size ** 3)
        self.fc_label = nn.Sequential(
            nn.Linear(final_dim // 2, self.n_label_class),
            nn.Softmax(),
        )
        self.fc_year = nn.Sequential(
            nn.Linear(final_dim - final_dim // 2, self.n_year_class),
            nn.Softmax(),
        )

    def forward(self, x):
        spec = self.spec_converter(x) # (batch_size, channels, n_mels, time_frames)

        # # Apply 1D batch normalization
        # # TODO(minigb): Use batchnorm?
        # # TODO(minigb): Check whether using 'view' here is appropriate
        # spec_reshaped = spec.view(spec.shape[0], -1, spec.shape[-1]) # (batch_size, n_in_channel * n_mels, time_frames)
        # spec = self.batchnorm1d(spec_reshaped).view(spec.shape)

        out = self.conv_layer(spec) # (batch_size, n_out_channel, n_mels // 27, time_frames // 27)
        out = out.flatten(1, 2) # (batch_size, n_out_channel * n_mels // 27, time_frames // 27)
        out = out.mean(dim=-1) # (batch_size, n_out_channel * n_mels // 27)

        label_output = self.fc_label(out[:, :out.size(1) // 2])
        year_output = self.fc_year(out[:, out.size(1) // 2:])

        return label_output, year_output
    

# @hydra.main(config_path='config', config_name='packed')
# def rum_dummy(cfg):
#     model = AudioModel(cfg.model.cfg)

#     batch_size = 1
#     n_in_channel = 1
#     x = torch.randn(batch_size, n_in_channel, cfg.model.cfg.sr * 30) # (batch_size, channels, time_frames)
#     out = model(x)
#     print(out.shape)

# if __name__ == '__main__':
#     rum_dummy()