import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import datetime

import model_zoo
from dataset import KpopDataset
from trainer import validate  # Assuming validate function is in trainer.py

@hydra.main(config_path='config', config_name='packed')
def main(cfg: DictConfig):
    device = cfg.train.device

    test_dataset = KpopDataset(cfg, cfg.dict_key.test)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.valid_batch_size, shuffle=False)

    model_class = getattr(model_zoo, cfg.model.name)
    model = model_class(cfg.model.cfg).to(device)
    criterion = nn.CrossEntropyLoss()

    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # Perform validation on the test set
    with torch.no_grad():
        test_loss = validate(model, test_loader, criterion, device, scaler=None, smoothing=cfg.train.smoothing, normalize=True)

    print(f'Test Loss: {test_loss:.4f}')
    wandb.run.summary["Test Loss"] = test_loss

    # Save the final model if needed
    torch.save(model.state_dict(), 'final_model.pth')
    wandb.save('final_model.pth')
    wandb.finish()

if __name__ == '__main__':
    main()
