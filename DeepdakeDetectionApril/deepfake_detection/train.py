import torch
import time
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#=====================
# model training
#=====================

def train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                num_epochs=10, scheduler=None, save_path=None):
    """
    train a model while tracking performance metrics
    
    args:
        model: pytorch model to train
        train_loader: dataloader for training data
        val_loader: dataloader for validation data
        criterion: loss function (usually crossentropy)
        optimizer: optimizer for updating weights
        device: device to train on (cuda/cpu)
        num_epochs: how many epochs to train for
        scheduler: learning rate scheduler (optional)
        save_path: where to save the best model (optional)
        
    returns:
        history: dictionary with training metrics
    """
    model = model.to(device)
    
    # initialize tracking variables
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    epoch_times = []
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        #=====================
        # training phase
        #=====================
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f'epoch {epoch+1}/{num_epochs} [train]')
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # zero the gradients
            optimizer.zero_grad()
            
            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # update progress bar
            train_bar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        #=====================
        # validation phase
        #=====================
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'epoch {epoch+1}/{num_epochs} [val]')
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # update progress bar
                val_bar.set_postfix({'loss': loss.item(), 'acc': 100 * val_correct / val_total})
                
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = 100 * val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)
        
        # step the scheduler if it exists
        if scheduler is not None:
            scheduler.step()
            
        # calculate epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # print epoch summary
        print(f'epoch {epoch+1}/{num_epochs}: '
              f'train loss: {epoch_loss:.4f}, train acc: {epoch_acc:.2f}%, '
              f'val loss: {val_epoch_loss:.4f}, val acc: {val_epoch_acc:.2f}%, '
              f'time: {epoch_time:.2f}s')
        
        #=====================
        # model saving
        #=====================
        # save best model based on validation accuracy
        if save_path is not None and val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), save_path)
            print(f'model saved to {save_path}')
    
    # if we saved the best model, load it back
    if save_path is not None:
        model.load_state_dict(torch.load(save_path))
    
    # return training history for visualization
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs,
        'epoch_times': epoch_times
    }
    
    return history

#=====================
# model evaluation
#=====================

def evaluate_model(model, test_loader, device):
    """
    evaluate a trained model on the test dataset
    
    args:
        model: trained pytorch model
        test_loader: dataloader for test data
        device: device to run evaluation on
        
    returns:
        results: dictionary with evaluation metrics
    """
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, target_names=['real', 'fake'])
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    print(f'test accuracy: {acc*100:.2f}%')
    print('\nclassification report:')
    print(class_report)
    print('\nconfusion matrix:')
    print(conf_matrix)
    
    return {
        'accuracy': acc,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix
    }