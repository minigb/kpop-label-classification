import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig

import model_zoo
from dataset import KpopDataset
from trainer import train_model

@hydra.main(config_path='config', config_name='packed')
def main(cfg: DictConfig):
    device = cfg.model.cfg.device

    num_samples = 1000
    inputs = torch.randn(num_samples, cfg.model.n_in_channel, cfg.model.sr * 30)  # Example input tensor
    labels = torch.randint(0, cfg.model.n_label_class, (num_samples,))  # Example label tensor
    years = torch.randint(0, cfg.model.n_year_class, (num_samples,))  # Example year tensor

    train_dataset = KpopDataset(cfg, cfg.dict_key.train)
    val_dataset = KpopDataset(cfg, cfg.dict_key.valid)
    # test_dataset = KpopDataset(cfg, cfg.dict_key.test)

    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    model_class = getattr(model_zoo, cfg.model.name)
    model = model_class(cfg.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    train_model(model, train_loader, val_loader, criterion, optimizer, cfg.num_epochs, device)

if __name__ == '__main__':
    main()
