import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import random
from ResNet18.models.resnet18 import ResNet, BasicBlock
from ResNet18.infrastructure.training_utils import train, validate, resnet, save_model, load_model, load_model_params
from ResNet18.infrastructure.utils import save_plots, get_data
from torchvision.models import ResNet18_Weights

# Testing load_model method
batch_size = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
train_loader, valid_loader = get_data(batch_size=batch_size)

# Loss function.
criterion = nn.CrossEntropyLoss()

# For testing purpose
resnet = resnet(
    weights=ResNet18_Weights.DEFAULT,
    img_channels=3, 
    num_layers=18,
    block=BasicBlock,
    num_classes=10, 
    device=device
    )

PATH = "/home/yuquand/ResNet18/CIFAIR10_trained_ResNet.pt"
load_model_params(resnet, PATH)

valid_epoch_loss, valid_epoch_acc = validate(
            resnet,
            valid_loader,
            criterion,
            device
        )

print(f"Validation loss: {valid_epoch_loss: .3f}, validation acc: {valid_epoch_acc: .3f}")

