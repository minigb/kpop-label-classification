import torch
from tqdm.auto import tqdm
import wandb

def smooth_labels(labels, num_classes, smoothing=0.1):
    """
    Apply label smoothing to the given labels.
    
    Args:
    - labels (torch.Tensor): Tensor of class indices.
    - num_classes (int): Total number of classes.
    - smoothing (float): Smoothing factor.

    Returns:
    - smoothed_labels (torch.Tensor): Smoothed label distributions.
    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    smooth_value = smoothing / (num_classes - 1)
    
    smoothed_labels = torch.full(size=(labels.size(0), num_classes), fill_value=smooth_value).to(labels.device)
    smoothed_labels.scatter_(1, labels.unsqueeze(1), confidence)
    
    return smoothed_labels

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, smoothing=0.1):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        inputs, labels, years = batch
        inputs, labels, years = inputs.to(device), labels.to(device), years.to(device)

        optimizer.zero_grad()
        label_output, year_output = model(inputs)

        label_loss = criterion(label_output, labels)
        
        # Apply label smoothing for year classification
        smoothed_years = smooth_labels(years, model.n_year_class, smoothing)
        year_loss = torch.mean(torch.sum(-smoothed_years * torch.log_softmax(year_output, dim=1), dim=1))
        
        loss = label_loss + year_loss

        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        running_loss += loss_value
        wandb.log({"Train Loss per Iteration": loss_value})

    average_loss = running_loss / len(dataloader)
    return average_loss

def validate_per_frequency(model, dataloader, criterion, device, smoothing=0.1):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels, years = batch
            inputs, labels, years = inputs.to(device), labels.to(device), years.to(device)

            label_output, year_output = model(inputs)

            label_loss = criterion(label_output, labels)

            # Apply label smoothing for year classification
            smoothed_years = smooth_labels(years, model.n_year_class, smoothing)
            year_loss = torch.mean(torch.sum(-smoothed_years * torch.log_softmax(year_output, dim=1), dim=1))

            loss = label_loss + year_loss

            loss_value = loss.item()
            running_loss += loss_value
            wandb.log({"Validation Loss per Period": loss_value})

    average_loss = running_loss / len(dataloader)
    return average_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, smoothing=0.1, valid_freq=1):
    best_loss = float('inf')
    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, smoothing)
        if epoch % valid_freq == 0: # currently valid_freq is set to 1
            val_loss = validate_per_frequency(model, val_loader, criterion, device, smoothing)

            if val_loss < best_loss:
                best_loss = val_loss
                # torch.save(model.state_dict(), 'best_model.pth')
                wandb.save('best_model.pth')
                wandb.run.summary["Best Validation Loss"] = best_loss

        # # Save the model at the end of every epoch
        # torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
        # wandb.save(f'model_epoch_{epoch+1}.pth')

    print('Training complete. Best validation loss:', best_loss)
    wandb.run.summary["Final Best Validation Loss"] = best_loss
