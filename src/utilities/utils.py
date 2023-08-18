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

################################
# Methods for data preprocessing.
################################

def get_a1_dataset(data_dir: str="/data/yuquand/a1_demo/") -> dict:
    """
    Parameters: 
    data_dir(str): The directory of a1 dataset where the images are arranged
    in this way by default:

    data_dir/part_x/image_AAAAA_BBBB_CCCC_DDDD.png

    where AAAAA is the Frame order, BBBB is vel_x, CCCC is vel_y, and DDDD is
    steering angle.

    Returns:
    dict: A dictonary containing tuples (img_data, label) with corresponding key
    equals to its batch dataset directory(e.g. part_1).
    """ 
    a1_datasets = {x: get_batch_dataset(os.path.join(data_dir, x)) for x in ['part_1', 'part_2']}

    return a1_datasets

def get_batch_dataset(directory: str):
    """
    Load batch dataset from the given directory. 

    Parameters:     
    directory(str): The directory of the batch dataset.

    Returns:
    tuple: (img_data, cmd_label) where both imag_data and cmd_label
           are numpy arrays.
    """
    print("-" * 100) 

     # Load file path and sort based on the order of the images.
    filenames = os.listdir(directory)
    filenames = sorted(filenames)

    # Parse images and labels respectively.
    images, labels = [], []
    for filename in filenames:
        f = os.path.join(directory, filename)

        # Parse image
        img = Image.open(f)
        images.append(np.array(img))

        # Parse label (vel_x, vel_y, theta)
        prefix = 'image_'
        suffix = '.png'
        cmd = filename[len(prefix):-len(suffix)].split("_")[1:]
        
        # Delete the vel_y column and only keep (vel_x, theta)
        cmd = np.delete(np.asarray(cmd, dtype=np.float64), 1, 0)
        labels.append(cmd)

    images = np.array(images)
    # N, H, W, C = images.shape
    # images = images.reshape((N, C, H, W)) # reshape into [N, C, H, W]
    images = images.transpose(0, 3, 1, 2)
    labels = np.array(labels)

    print(f"images shape: {images.shape}")
    print(f"labels shape: {labels.shape}\n")
    print(f"Finished loading a1 batch dataset from directory {directory}.")
    print("-" * 100)

    return images, labels

def extract_batch_feature(device, model, batch_data, batch_size:int=128):
    """
    Extract features from the raw images but keep the order
    of the image and command pairs.

    Return 
    """
    model.eval()
    input_feature, label_feature = [], []
    batch_inputs, batch_labels = batch_data

    val_transforms =  transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = CustomTensorDataset(
        tensors=(torch.tensor(batch_inputs), torch.tensor(batch_labels)),
        transform=val_transforms
    )

    # Set shuffle to false for keeping the order of images.
    data_loader = DataLoader(
        dataset=dataset,
        shuffle=False, 
        batch_size=batch_size
    )

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            input, label = data
            input = input.to(device)
            label = label.to(device)
            output = model(input)

            # Reshape Batch_dim xFeature_dim x 1 x 1 to Batch_dim xFeature_dim.
            batch_dim, feature_dim = output.size()[0:2]
            input_feature.append(output.reshape((batch_dim, feature_dim))) 
            label_feature.append(label)

    # Convert tensors to Numpy array.
    input_feature = torch.cat(input_feature, dim=0).detach().cpu().numpy()
    label_feature = torch.cat(label_feature, dim=0).detach().cpu().numpy()

    # TODO: comment it out for debugging
    # print("-"*100)
    # print(f"inputs feature shape: {input_feature.shape}")
    # print(f"labels shape: {label_feature.shape}")
    # print("Finished extract feature from current batch data.")
    # print("-"*100)

    return input_feature, label_feature

class CustomTensorDataset(Dataset):
    """
    TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
    
    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y
    
    def __len__(self):
        return self.tensors[0].size(0)

def get_a1_dataloader(feature_dataset, batch_size: int, interval: int, train_ratio:float=0.7, ):
    """
    Given feature_datatset, feed data in pretrained ResNet50 and split dataset
    as train, val dataloader.
    """
    imgs, cmd_labels = concat_imgs(feature_dataset, interval)
    
    # Random Split into train and validation set.
    train_size = int(np.floor(imgs.shape[0] * train_ratio))
    print(f"training set size: {train_size}, validation set size: {imgs.shape[0]-train_size}")
    print("-"*100)

    dataset = TensorDataset(torch.tensor(imgs), torch.tensor(cmd_labels))
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, imgs.shape[0] - train_size])
    train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(dataset=val_set, shuffle=True, batch_size=batch_size)

    return train_loader, val_loader

def concat_imgs(feature_dataset, interval):
    """
    Concatenate the original images with their goal images by the given
    interval.

    Return a pair of concatenated images features and their lables as a
    tuple of numpy arrays with size ((N, 4096), (N, 2)) where 4096 is
    the number of concatenated features.
    """
    # Concatenate start and goal image.
    imgs, cmd_labels = [], []
    for batch_dir in feature_dataset:
        img_data, cmd_label = feature_dataset[batch_dir]
        
        original_img = img_data[:-interval, :]
        goal_img = img_data[interval:, :]
        cat_img = np.concatenate((original_img, goal_img), axis=-1)
        # interval+1 for including the goal image.
        cat_cmd = average_cmd(cmd_label, interval+1)  # Take the average command over the interval.

        imgs.append(cat_img)
        cmd_labels.append(cat_cmd)

        print(f"interval: {interval}")
        print(f"[concatenated] image data size : {cat_img.shape}")
        print(f"[concatenated] cmd_label size : {cat_cmd.shape}")
        print("-"*100)
    
    imgs = np.concatenate(imgs)
    cmd_labels = np.concatenate(cmd_labels)

    return imgs, cmd_labels

def average_cmd(cmd_label, interval):
    cmd_label = cmd_label
    vel_x = cmd_label[:, 0]
    steering_angle = cmd_label[:, 1]
    print("-"*100)
    print(f"vel_x size: {vel_x.shape}, steering_angle size:{steering_angle.shape}")

    kernel = np.full((interval,), 1)
    vel_x_avg = np.convolve(vel_x, kernel, mode='valid') / len(kernel)
    steering_angle_avg = np.convolve(steering_angle, kernel, mode='valid') / len(kernel)

    cat_cmd = np.stack((vel_x_avg, steering_angle_avg), axis=1)
    print(f"[before convoluted] cmd_label size: {cmd_label.shape}")
    print(f"[after convoluted] cmd_label size: {cat_cmd.shape}")
    print("-"*100)
    return torch.from_numpy(cat_cmd).float()

############################################################
# Other utils methods(plot data, save and load dataset, etc)
############################################################
def pretrained_ResNet50(device):
    # Initailize ResNet50 as Pretrain model and treat it as fix feature extractor
    resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    modules = list(resnet50.children())[:-1]
    resnet50 = nn.Sequential(*modules)
    for p in resnet50.parameters():
        p.requires_grad = False
    resnet50.to(device)
    resnet50.eval()
    print("Set pretrained resnet50 into evaluation mode.")
    return resnet50

def save(file, filepath:str) -> None:
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filepath, 'wb') as fp:
        pickle.dump(file, fp)
    print(f"Saved as {filepath}")
    print("-"*100)

def load(filepath:str):
    file = pickle.load(open(filepath, "rb"))
    print(f"Loaded file {filepath}")
    print("-"*100)
    return file

def log_metrics_figure(train_result, valid_result, path: str='../images_outputs', name: str=None, loss: bool=True):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # Create the directory if it doesn't exist.
    if not os.path.exists(path):
        os.makedirs(path)

    train_label = 'train loss' if loss else 'train accuracy'
    validation_label = 'validataion loss' if loss else 'validataion accuracy'
    xlabel = 'Epochs'
    ylabel = 'Loss' if loss else 'Accuracy'
    name = name if name is not None else ('loss' if loss else 'accuracy')

    # Plot figure.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_result, color='tab:blue', linestyle='-',
        label=train_label
    )
    plt.plot(
        valid_result, color='tab:red', linestyle='-',
        label=validation_label
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(os.path.join(path, name + '.png'))
    print(f"Successfully saved {os.path.join(path, name + '.png')}")





