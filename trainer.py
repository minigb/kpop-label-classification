import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
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

    average_loss = running_loss / len(dataloader)
    wandb.log({"Train Loss": average_loss, "Epoch": epoch})
    return average_loss

def validate_one_epoch(model, dataloader, criterion, device, epoch):
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

    average_loss = running_loss / len(dataloader)
    wandb.log({"Validation Loss": average_loss, "Epoch": epoch})
    return average_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_loss = float('inf')
    for epoch in tqdm(range(num_epochs), desc='Running Epochs'):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = validate_one_epoch(model, val_loader, criterion, device, epoch)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            wandb.run.summary["Best Validation Loss"] = best_loss

    print('Training complete. Best validation loss:', best_loss)
    wandb.run.summary["Final Best Validation Loss"] = best_loss
