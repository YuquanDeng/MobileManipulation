import torch
from tqdm import tqdm
import sys
sys.path.append('../models')
from typing import Any, Type, Dict, TypeVar
import numpy as np
import matplotlib.pyplot as plt
import os

def train(model, trainloader, optimizer, criterion, device, reg:bool=False):
    """
    Train policy for one epoch. Return epoch loss and epoch accuracy. 
    """
    model.train()
    print('Training')

    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0

    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, labels)

        # Calculate loss and Accuracy
        train_running_loss += loss.item()

        if not reg:
            _, preds = torch.max(outputs, dim=1)  # return (max_value, argmax)
            train_running_correct += (preds == labels).sum().item()

        # Back Prop + Update weight
        loss.backward()
        optimizer.step()

    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))

    return epoch_loss, epoch_acc

def validate(model, testloader, criterion, device, reg:bool=False):
    """
    Evaluate the policy for one epoch. Return epoch loss and epoch accuracy. 
    """
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0

    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)

            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()

            if not reg:
                # Calculate the accuracy.
                _, preds = torch.max(outputs.data, 1)
                valid_running_correct += (preds == labels).sum().item()

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset)) if not reg else 0 # will be 0 if reg flag is True.
    return epoch_loss, epoch_acc

def save_model_params(model, path: str):
    torch.save(model.state_dict(), path)
    print('-'*100)
    print(f"Successfully saved model as {path}")

def load_model_params(model, path: str):
    model.load_state_dict(torch.load(path))
    print('-'*100)
    print(f"Successfully loaded model as {path}")

def save_model(model, path: str):
    torch.save(model, path)
    print('-'*100)
    print(f"Successfully saved model as {path}")

def load_model(path: str):
    print('-'*100)
    print(f"Successfully loaded model from {path}")
    return torch.load(path)
 
def save_prediction_plots(model, testloader, device, val_data: bool, path:str='../images_outputs'):
    # Plot validation Scatter graph.
    model.eval()
    vel_x_plot_name = 'vel_x_regression_val.png' if val_data else 'vel_x_regression_train.png'
    steering_angle_plot_name = 'steering_angle_regression_val.png' if val_data else 'steering_angle_regression_train.png'
    predict_vel_x, true_vel_x = [], []
    predict_theta, true_theta = [], []   

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)

            predict_vel_x.append(outputs[:, 0])
            true_vel_x.append(labels[:, 0])
            predict_theta.append(outputs[:, 1])
            true_theta.append(labels[:, 1])
    
    predict_vel_x = torch.cat(predict_vel_x, dim=0)
    predict_theta = torch.cat(predict_theta, dim=0)
    true_vel_x = torch.cat(true_vel_x, dim=0)
    true_theta = torch.cat(true_theta, dim=0)

    plt.figure(figsize=(10, 7))
    plt.scatter(true_vel_x.cpu().data.numpy(), predict_vel_x.cpu().data.numpy(), alpha=0.5)
    plt.xlabel('True vel_x')
    plt.ylabel('Predict vel_x')
    plt.savefig(os.path.join(path, vel_x_plot_name))
    print(f"Successfully saved {os.path.join(path, vel_x_plot_name)}")

    plt.figure(figsize=(10, 7))
    plt.scatter(true_theta.cpu().data.numpy(), predict_theta.cpu().data.numpy(), c="y", alpha=0.5)
    plt.xlabel('True steering angle')
    plt.ylabel('Predict steering angle')
    plt.savefig(os.path.join(path, steering_angle_plot_name))
    print(f"Successfully saved {os.path.join(path, steering_angle_plot_name)}")
