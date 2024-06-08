import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import hydra
from omegaconf import DictConfig

from model_zoo import AudioModel
from trainer import Trainer

@hydra.main(config_path='config', config_name='packed')
def main(cfg: DictConfig):
    device = cfg.train.device

    # Dummy dataset for demonstration purposes
    num_samples = 1000
    inputs = torch.randn(num_samples, cfg.model.n_in_channel, cfg.model.sr * 30)  # Example input tensor
    labels = torch.randint(0, cfg.model.n_label_class, (num_samples,))  # Example label tensor
    years = torch.randint(0, cfg.model.n_year_class, (num_samples,))  # Example year tensor

    dataset = TensorDataset(inputs, labels, years)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    model = AudioModel(cfg.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, cfg.num_epochs, device)
    trainer.train()

if __name__ == '__main__':
    main()
