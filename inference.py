import torch
import torchaudio
from omegaconf import DictConfig, OmegaConf
import hydra

import model_zoo
from module import SpecModel  # Assuming SpecModel is in module.py

@hydra.main(config_path='config', config_name='packed')
def main(cfg: DictConfig):
    device = cfg.train.device

    model_class = getattr(model_zoo, cfg.model.name)
    model = model_class(cfg.model.cfg).to(device)

    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # Example new data for inference (replace with actual data)
    example_audio_file = 'path_to_new_audio_file.mp3'
    spec_converter = SpecModel(cfg.model.cfg.sr, cfg.model.cfg.n_fft, cfg.model.cfg.hop_length, cfg.model.cfg.n_mels)
    
    # Load and process the audio file
    waveform, sr = torchaudio.load(example_audio_file)
    if sr != cfg.model.cfg.sr:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=cfg.model.cfg.sr)(waveform)
    
    # Convert to spectrogram
    spec = spec_converter(waveform.unsqueeze(0))  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        spec = spec.to(device)
        label_output, year_output = model(spec)
        predicted_label = torch.argmax(label_output, dim=1).item()
        predicted_year = torch.argmax(year_output, dim=1).item()

    # Decode predictions if necessary
    label_decoder = {v: k for k, v in cfg.label_dict.items()}
    year_decoder = {v: k for k, v in enumerate(range(1994, 2025))}

    print(f'Predicted Label: {label_decoder[predicted_label]}')
    print(f'Predicted Year: {year_decoder[predicted_year]}')

if __name__ == '__main__':
    main()
