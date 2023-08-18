import torch
import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
sys.path.append("../")
from tqdm import tqdm
from infrastructure.utils import pretrained_ResNet50, load, get_a1_dataset
from infrastructure.training_utils import load_model, load_model_params, save_model
from policies.MLP_policy import MLP_policy
from scripts.run_experiment import MLP_trainer
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms


"""
This file used for testing the pretrained MLP Policy.
"""

def test_policy(path:None):
    # Part 1: 
    # (1) Load raw dataset and randomly sample 9 images with labels;
    # Plot the images first. 
    a1_dataset = load('a1_dataset')

    batch_name = 'part_1'
    n = len(a1_dataset[batch_name][0])
    sample_index  = np.random.randint(0, n, 9)
    samples = a1_dataset[batch_name][0][sample_index]
    samples_labels = a1_dataset[batch_name][1][sample_index]

    # Sanity Check
    log_data(sample_index, False)
    log_data(samples, True)
    log_data(samples_labels, True)

    # Plot raw imgs.
    log_figure(samples, samples_labels, name='a1_data_raw_imgs_plot', path=path)

    # Part 2: Augment the sample images and input into ResNet 50 for
    # Extracting features. Then input the features into pretrained Policy
    # and get actions.
    device = torch.device("cpu")
    curr_dir = os.getcwd()

    # Load Policy differently.
    # policy = load_model(os.path.join(curr_dir, "../models/MLP_policy.pt"))
    # policy.to(device)
    input_size = 4096
    policy = MLP_policy(input_size=input_size, size=4, output_size=2) # Initalize as a random policy 
    load_model_params(policy, os.path.join(curr_dir, path + '/MLP_policy.pt'))
    resnet50 = pretrained_ResNet50(device)

    interval = 6
    goal_imgs = a1_dataset[batch_name][0][sample_index + interval] 
    goal_imgs_labels = a1_dataset[batch_name][1][sample_index + interval] 

    log_figure(goal_imgs, goal_imgs_labels, name='a1_data_goal_imgs_plot', path=path)

    actions = []
    for i in range(9):
        curr_img = samples[i, :]
        goal_img = goal_imgs[i, :]
        action = get_policy(policy, resnet50, curr_img, goal_img, device)
        actions.append(action)
    
    log_figure(samples, np.array(actions), name='a1_data_raw_imgs_prediction_plot', path=path)

def get_policy(MLP_policy, resnet50, curr_img, goal_img, device):
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    resnet50.eval()
    MLP_policy.eval()
    with torch.no_grad():
        # Extract features.
        # Change into HxWxC
        C, H, W = curr_img.shape
        curr_img = curr_img.reshape((H, W, C))
        goal_img = goal_img.reshape((H, W, C))

        curr_img = val_transform(curr_img).unsqueeze(0).to(device)
        goal_img = val_transform(goal_img).unsqueeze(0).to(device)
        curr_features = resnet50(curr_img)
        goal_features = resnet50(goal_img)
        print(f'curr_features size: {curr_features.size()}')
        combined_features = torch.cat((curr_features.reshape((1, 2048)), goal_features.reshape((1, 2048))), -1)
        log_data(combined_features, False, 'combined_features')

        # Output actions.
        combined_features.to(device)
        action = MLP_policy(combined_features)

    print(f"action: {action}")
    return action.detach().cpu().numpy()


    
def log_data(data, shape: bool, name: str='data'):
    if shape:
        print(f"{name} shape: {data.shape}\n")
    else:
        print(f"{name}: {data}\n")

def log_figure(images, labels, name: None, path:str):
    rows = 3
    columns = 3
    fig = plt.figure(figsize=(10, 8))
    for i in range(9):
        fig.add_subplot(rows, columns, i+1)
        # Change into HxWxC.
        C, H, W = images[i, :].shape
        curr_img = images[i, :].reshape((H, W, C))
        plt.imshow(curr_img)
        plt.title(f"{labels[i, :]}")
    plt.suptitle(name)
    plt.savefig(os.path.join(path,  f'{name}.png'))

def save_ResNet50():
    resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
    curr_directory = os.getcwd()
    save_model(resnet50, path=curr_directory + '/test_model/resnet_50.pt')

def resnet50_inference():
    from PIL import Image
    import json
    import requests
    import warnings
    warnings.filterwarnings('ignore')
    
    device = torch.device('cpu')
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')
    curr_directory = os.getcwd()
    resnet50 = load_model(curr_directory + '/test_model/resnet_50.pt')
    resnet50.eval().to(device)

    uris = [
        'http://images.cocodataset.org/test-stuff2017/000000024309.jpg',
        'http://images.cocodataset.org/test-stuff2017/000000028117.jpg',
        'http://images.cocodataset.org/test-stuff2017/000000006149.jpg',
        'http://images.cocodataset.org/test-stuff2017/000000004954.jpg',
    ]

    batch = torch.cat(
        [utils.prepare_input_from_uri(uri) for uri in uris]
    ).to(device)

    print(f"batch size: {batch.size()}")

    with torch.no_grad():
        output = torch.nn.functional.softmax(resnet50(batch), dim=1)
        
    results = utils.pick_n_best(predictions=output, n=5)

    counter = 0
    for uri, result in zip(uris, results):
        img = Image.open(requests.get(uri, stream=True).raw)
        img.thumbnail((256,256), Image.ANTIALIAS)
        plt.imshow(img)
        plt.savefig(curr_directory + f'/bad_plots/img_{counter}.png')
        counter += 1
        print(result)


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--path_key', '-p', type=str, required=True)
    # args = parser.parse_args()
    # params = vars(args)     # convert args to dictionary

    arg_path= sys.argv[1]
    split_path = [arg_path[i:i+2] for i in range(0, len(arg_path), 2)]
    path = f'../conf_outputs/2023-{split_path[0]}-{split_path[1]}/{split_path[2]}-{split_path[3]}'
    test_policy(path=path)

