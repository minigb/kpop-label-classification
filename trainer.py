import torch
from tqdm.auto import tqdm
import wandb

def smooth_labels(labels, num_classes, smoothing):
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    smooth_value = smoothing / 2  # Since we distribute the smoothing to two adjacent classes
    
    # Initialize the smoothed labels tensor with zeros
    smoothed_labels = torch.zeros(size=(labels.size(0), num_classes), device=labels.device)
    
    # Assign confidence to the true class
    smoothed_labels.scatter_(1, labels.unsqueeze(1), confidence)
    
    # Assign smooth_value to the adjacent classes
    for i in range(labels.size(0)):
        if labels[i] > 0:  # If not the first class
            smoothed_labels[i, labels[i] - 1] = smooth_value
        if labels[i] < num_classes - 1:  # If not the last class
            smoothed_labels[i, labels[i] + 1] = smooth_value
    
    return smoothed_labels

def get_loss(run_type, criterion, label_output, year_output, labels, years, smoothing):
    label_loss = criterion(label_output, labels)
    
    # Apply label smoothing for year classification
    smoothed_years = smooth_labels(years, year_output.size(1), smoothing)
    year_loss = torch.mean(torch.sum(-smoothed_years * torch.log_softmax(year_output, dim=1), dim=1))
    wandb.log({f'[{run_type}] Label Loss': label_loss.item(),
               f'[{run_type}] Year Loss': year_loss.item()})
    
    loss = (label_loss + year_loss) / 2
    return loss

def train_one_epoch(model, dataloader, criterion, optimizer, device, smoothing):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        inputs, labels, years = batch
        inputs, labels, years = inputs.to(device), labels.to(device), years.to(device)

        optimizer.zero_grad()
        label_output, year_output = model(inputs)

        loss = get_loss('Train', criterion, label_output, year_output, labels, years, smoothing)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        running_loss += loss_value
        wandb.log({"Train Loss per Iteration": loss_value})

    average_loss = running_loss / len(dataloader)
    return average_loss

def validate(model, dataloader, criterion, device, smoothing):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels, years = batch
            inputs, labels, years = inputs.to(device), labels.to(device), years.to(device)

            label_output, year_output = model(inputs)

            loss = get_loss('Valid', criterion, label_output, year_output, labels, years, smoothing)
            loss_value = loss.item()
            running_loss += loss_value
            wandb.log({"Validation Loss per Period": loss_value})

    average_loss = running_loss / len(dataloader)
    return average_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, smoothing, valid_freq):
    best_loss = float('inf')
    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        train_one_epoch(model, train_loader, criterion, optimizer, device, smoothing)
        if epoch % valid_freq == 0: # currently valid_freq is set to 1
            val_loss = validate(model, val_loader, criterion, device, smoothing)

            if val_loss < best_loss:
                best_loss = val_loss
                wandb.save('best_model.pth')
                wandb.run.summary["Best Validation Loss"] = best_loss

    print('Training complete. Best validation loss:', best_loss)
    wandb.run.summary["Final Best Validation Loss"] = best_loss
