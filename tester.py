import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import hydra
from omegaconf import OmegaConf

import model_zoo
from dataset import KpopDataset
from trainer import validate_with_accuracy  # Assuming validate_with_accuracy function is in trainer.py

@hydra.main(config_path='config', config_name='packed')
def main(cfg):
    device = cfg.train.device

    best_model_config_path = Path(cfg.test.best_model_config_path)
    best_model_config = OmegaConf.load(best_model_config_path)
    # model_config.n_in_channel = 2 # TODO(minigb): Don't know why but there's an error
    model_class = getattr(model_zoo, best_model_config.name)
    model = model_class(best_model_config.cfg).to(device)
    criterion = nn.CrossEntropyLoss()

    # Load the best model
    best_model_pt_path = Path(cfg.test.best_model_pt_path)
    model.load_state_dict(torch.load(best_model_pt_path))
    model.eval()

    dataset_mode_list = ['train', 'valid', 'test']
    for dataset_mode in dataset_mode_list:
        test_dataset = KpopDataset(cfg, dataset_mode)
        test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False)

        # Perform validation on the test set
        with torch.no_grad():
            test_loss, label_accuracy, year_accuracy, total_examples = validate_with_accuracy(
                model, test_loader, criterion, device, smoothing=cfg.train.smoothing, normalize=True
            )

        print(f'Test Loss: {test_loss:.4f}')
        print(f'Test Label Accuracy: {label_accuracy:.4f}')
        print(f'Test Year Accuracy: {year_accuracy:.4f}')
        print(f'Total Examples: {total_examples}')


if __name__ == '__main__':
    main()
