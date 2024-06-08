import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class AudioDataset(Dataset):
    # Dummy dataset class, replace with your actual dataset class
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class Trainer:
    def __init__(self, model, device, train_loader, val_loader, criterion, optimizer, num_epochs):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            label_output, year_output = self.model(inputs)
            loss = self.criterion(label_output, labels[:, 0]) + self.criterion(year_output, labels[:, 1])

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(self.train_loader)
        print(f'Epoch {epoch+1}/{self.num_epochs}, Training Loss: {epoch_loss:.4f}')

    def validate_one_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                label_output, year_output = self.model(inputs)
                loss = self.criterion(label_output, labels[:, 0]) + self.criterion(year_output, labels[:, 1])

                running_loss += loss.item()

        epoch_loss = running_loss / len(self.val_loader)
        print(f'Epoch {epoch+1}/{self.num_epochs}, Validation Loss: {epoch_loss:.4f}')

    def train(self):
        for epoch in range(self.num_epochs):
            self.train_one_epoch(epoch)
            self.validate_one_epoch(epoch)
            self.save_model(epoch)

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), f'model_epoch_{epoch+1}.pth')
        print(f'Model saved at epoch {epoch+1}')

