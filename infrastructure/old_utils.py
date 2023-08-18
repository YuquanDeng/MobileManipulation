"""
Methods that may no longer be useful.
"""
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import numpy as np
from PIL import Image
import pickle
import torch.nn as nn

def get_data(batch_size=64):
    data_train = datasets.CIFAR10(
        root='~/data',
        download=True,
        train=True,
        transform=ToTensor()
    )


    data_val = datasets.CIFAR10(
        root='~/data',
        download=True,
        train=False,
        transform=ToTensor()
    )

    train_loader = DataLoader(data_train, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(data_val, shuffle=False, batch_size=batch_size)
    return train_loader, val_loader

def plot_control_law(cmd_label: np.array, name=None) -> None:
    forward_speed = cmd_label[:, 0]
    side_speed = cmd_label[:, 1]
    steering_angle = cmd_label[:, 2]

    plt.figure(figsize=(10, 7))
    plt.plot(
        forward_speed, color='tab:blue', linestyle='-',
        label='forward_speed(vel_x)'
    )
    plt.plot(
        side_speed, color='tab:red', linestyle='-',
        label='side_speed(vel_y)'
    )

    plt.xlabel('Time(t)')
    plt.ylabel('Speed(v)')
    plt.legend()
    plt.savefig(os.path.join('outputs', name + '_speed.png'))

    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        steering_angle, color='tab:blue', linestyle='-',
        label='steering_angle'
    )

    plt.xlabel('Time(t)')
    plt.ylabel(r'Steering angle($\theta$)')
    plt.legend()
    plt.savefig(os.path.join('outputs', name + '_steering_angle.png'))
