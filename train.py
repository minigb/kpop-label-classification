import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import datetime

import model_zoo
from dataset import KpopDataset
from trainer import train_model

@hydra.main(config_path='config', config_name='packed')
def main(cfg: DictConfig):
    device = cfg.train.device

    # Use KpopDataset for train and validation datasets
    train_dataset = KpopDataset(cfg, cfg.dict_key.train)
    val_dataset = KpopDataset(cfg, cfg.dict_key.valid)

    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    model_class = getattr(model_zoo, cfg.model.name)
    model = model_class(cfg.model.cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    run_name = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{cfg.wandb.run_name}'
    wandb.init(project=cfg.wandb.project_name,
               config=OmegaConf.to_container(cfg, resolve=True),
               name=run_name)
    # # Watch the model for gradient updates, logging every 1000 batches
    # wandb.watch(model, log="gradients", log_freq=1000)

    # Train the model and log with wandb
    train_model(model, train_loader, val_loader, criterion, optimizer, cfg.train.num_epochs, valid_freq=cfg.train.valid_freq, smoothing=cfg.train.smoothing, device=device)

    # Save the model at the end of training
    # torch.save(model.state_dict(), 'final_model.pth')
    wandb.save('final_model.pth')

    # Finish the wandb run
    wandb.finish()

if __name__ == '__main__':
    main()
