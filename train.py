import time, copy

import torch

import matplotlib.pyplot as plt

from dataset import imshow


def train_model(model, dataloaders, criterion, optimizer, scheduler, 
                phases, dataset_sizes, device=None, num_epochs=25):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    for epoch in range(num_epochs):
        # display epoch
        print('Epoch {} / {}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        for phase in phases:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
                
            # reset loss for current phase and epoch
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                # track history only during training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    preds_onehot = torch.zeros_like(labels.data)
                    for i in range(preds_onehot.shape[1]):
                        preds_onehot[:, i, :, :] = preds == i
                    
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds_onehot == labels.data)
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() \
                        / (dataset_sizes[phase] * labels.data.shape[-2] * labels.data.shape[-1])
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))
    
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, dataloaders, labels, num_images=6):
    # save model training state to reset once finished
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    
    # without applying gradients
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # evaluate inputs
            outputs = model(inputs)
            # find argmax of class predictions
            _, preds = torch.max(outputs, 1)
            
            # for all images in inputs
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(labels[preds[j]]))
                imshow(inputs.cpu().data[j])
                
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
                
        model.train(mode=was_training)