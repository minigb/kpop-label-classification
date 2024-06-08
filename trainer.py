import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device
        self.best_loss = float('inf')

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        for batch in self.train_loader:
            inputs, labels, years = batch
            inputs, labels, years = inputs.to(self.device), labels.to(self.device), years.to(self.device)

            self.optimizer.zero_grad()
            label_output, year_output = self.model(inputs)

            label_loss = self.criterion(label_output, labels)
            year_loss = self.criterion(year_output, years)
            loss = label_loss + year_loss

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate_one_epoch(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, labels, years = batch
                inputs, labels, years = inputs.to(self.device), labels.to(self.device), years.to(self.device)

                label_output, year_output = self.model(inputs)

                label_loss = self.criterion(label_output, labels)
                year_loss = self.criterion(year_output, years)
                loss = label_loss + year_loss

                running_loss += loss.item()

        return running_loss / len(self.val_loader)

    def train(self):
        for epoch in range(self.num_epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.validate_one_epoch()

            print(f'Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')

        print('Training complete. Best validation loss:', self.best_loss)
