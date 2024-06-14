import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
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
    def _get_label_loss(label_output, labels):
        if label_output is None:
            return -1
        if wandb.run and run_type == 'Train':
            wandb.log({f'[{run_type}] Label Loss': label_loss.item()})
        return criterion(label_output, labels)
    
    def _get_year_loss(year_output, years):
    
        # Apply label smoothing for year classification
        smoothed_years = smooth_labels(years, year_output.size(1), smoothing)
        year_loss = torch.mean(torch.sum(-smoothed_years * torch.log_softmax(year_output, dim=1), dim=1))
        if wandb.run and run_type == 'Train':
            wandb.log({f'[{run_type}] Year Loss': year_loss.item()})

        return year_loss
    
    label_loss = _get_label_loss(label_output, labels)
    year_loss = _get_year_loss(year_output, years)

    if label_loss == -1:
        return year_loss
    return (label_loss + year_loss) / 2

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

        running_loss += loss.item()
        if wandb.run:
            wandb.log({"[Train] Loss per Iteration": loss.item()})

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
            running_loss += loss.item()

    average_loss = running_loss / len(dataloader)
    if wandb.run:
        wandb.log({"[Valid] Loss per Period": average_loss})
    return average_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, smoothing, valid_freq):
    best_loss = float('inf')
    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        train_one_epoch(model, train_loader, criterion, optimizer, device, smoothing)
        if epoch % valid_freq == 0:
            val_loss = validate(model, val_loader, criterion, device, smoothing)

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pth')
                if wandb.run:
                    wandb.save('best_model.pth')
                    wandb.run.summary["Best Validation Loss"] = best_loss

    print('Training complete. Best validation loss:', best_loss)
    wandb.run.summary["Final Best Validation Loss"] = best_loss

def validate_with_accuracy(model, dataloader, criterion, device, smoothing, normalize=True):
    model.eval()
    running_loss = 0.0
    correct_label_preds = 0
    correct_year_preds = 0
    total_examples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs, labels, years = batch
            inputs, labels, years = inputs.to(device), labels.to(device), years.to(device)

            label_output, year_output = model(inputs)
            loss = get_loss('Test', criterion, label_output, year_output, labels, years, smoothing)

            running_loss += loss.item()

            # Collect predictions and true labels for accuracy computation
            label_preds = torch.argmax(label_output, dim=1)
            year_preds = torch.argmax(year_output, dim=1)
            
            correct_label_preds += (label_preds == labels).sum().item()
            correct_year_preds += (year_preds == years).sum().item()
            total_examples += labels.size(0)

    average_loss = running_loss / len(dataloader)
    
    # Compute accuracy
    label_accuracy = correct_label_preds / total_examples
    year_accuracy = correct_year_preds / total_examples

    return average_loss, label_accuracy, year_accuracy, total_examples
