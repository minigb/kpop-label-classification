import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import hydra

import model_zoo
from dataset import KpopDataset
from trainer import validate_with_accuracy  # Assuming validate_with_accuracy function is in trainer.py

@hydra.main(config_path='config', config_name='packed')
def main(cfg):
    device = cfg.train.device

    test_dataset = KpopDataset(cfg, cfg.dict_key.test)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    best_model_files_path = Path(cfg.test.best_model_files_path)
    best_model_config_path = best_model_files_path / 'config.yaml'
    best_model_config = hydra.utils.instantiate(cfg, config_path=best_model_config_path)
    best_model_path = best_model_files_path / 'best_model.pth'

    model_config = best_model_config.model
    model_class = getattr(model_zoo, model_config.name)
    model = model_class(model_config.cfg).to(device)
    criterion = nn.CrossEntropyLoss()

    print(model)

    # Load the best model
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Perform validation on the test set
    with torch.no_grad():
        test_loss, label_accuracy, year_accuracy = validate_with_accuracy(
            model, test_loader, criterion, device, smoothing=cfg.train.smoothing, normalize=True
        )

    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Label Accuracy: {label_accuracy:.4f}')
    print(f'Test Year Accuracy: {year_accuracy:.4f}')


if __name__ == '__main__':
    main()
