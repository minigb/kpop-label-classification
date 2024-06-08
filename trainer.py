import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        inputs, labels, years = batch
        inputs, labels, years = inputs.to(device), labels.to(device), years.to(device)

        optimizer.zero_grad()
        label_output, year_output = model(inputs)

        label_loss = criterion(label_output, labels)
        year_loss = criterion(year_output, years)
        loss = label_loss + year_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels, years = batch
            inputs, labels, years = inputs.to(device), labels.to(device), years.to(device)

            label_output, year_output = model(inputs)

            label_loss = criterion(label_output, labels)
            year_loss = criterion(year_output, years)
            loss = label_loss + year_loss

            running_loss += loss.item()

    return running_loss / len(dataloader)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    print('Training complete. Best validation loss:', best_loss)
