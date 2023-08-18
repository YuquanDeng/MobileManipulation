import numpy as np
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
from keras.utils import load_img, img_to_array
import sys
import pickle
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import torch
from torch.utils.data import DataLoader, TensorDataset
import random
sys.path.insert(0, "../")

def save(file, name=None) -> None:
    global_dir = '/home/yuquand/ResNet18/data/'
    name = global_dir + name + ".pkl"
    with open(name, 'wb') as f:
        pickle.dump(file, f)
        print(f"{name} saved successfully.")
    print("-"*100)

def load(name=None) -> None:
    global_dir = '/home/yuquand/ResNet18/data/'
    name = global_dir + "a1_dataset.pkl" if name == None else global_dir + name + ".pkl"
    file = pickle.load(open(name, "rb"))
    
    print(f"{name} loaded successfully!")
    print("-"*100)

    return file

def get_mapping_class():
    # Parse Synset labels mapping.
    mapping_path = '../data/LOC_synset_mapping.txt'

    # Creation of mapping dictionaries to obtain the image classes
    class_mapping_dict = {}         # e.g. ('n01440764': 'tench, Tinca tinca')
    class_mapping_dict_number = {}  # e.g. (0: 'tench, Tinca tinca')
    mapping_class_to_number = {}    # e.g. ('n01440764': 0)
    mapping_number_to_class = {}    # e.g. (0: 'n01440764')
    i = 0
    for line in open(mapping_path):
        class_mapping_dict[line[:9].strip()] = line[9:].strip()
        class_mapping_dict_number[i] = line[9:].strip()
        mapping_class_to_number[line[:9].strip()] = i
        mapping_number_to_class[i] = line[:9].strip()
        i+=1

    print(f"class_mapping_dict: {class_mapping_dict}\n")
    print(f"class_mapping_dict_number: {class_mapping_dict_number}\n")
    print(f"mapping_class_to_number: {mapping_class_to_number}\n")
    print(f"mapping_number_to_class: {mapping_number_to_class}\n")

    return class_mapping_dict, class_mapping_dict_number, mapping_class_to_number, mapping_number_to_class

def get_imagenet():
    class_mapping_dict, class_mapping_dict_number, mapping_class_to_number, mapping_number_to_class = get_mapping_class()

    # Parse training Set data.
    train_path = '/data/yuquand/ILSVRC/Data/DET/train/ILSVRC2013_train'

    # Creation of dataset_array and true_classes
    true_classes = []
    images_array = []
    numbers_list = []
    for train_class in tqdm(os.listdir(train_path)):
        if train_class in class_mapping_dict:
            i = 0
            for el in os.listdir(train_path + '/' + train_class):
                if i < 10:
                    path = train_path + '/' + train_class + '/' + el
                    image = load_img(path,target_size=(224,224,3))

                    image_array = img_to_array(image).astype(np.uint8)
                    images_array.append(image_array)
                    true_class = class_mapping_dict[path.split('/')[-2]]
                    number = mapping_class_to_number[path.split('/')[-2]]
                    true_classes.append(true_class)
                    numbers_list.append(number)
                    i+=1
                else:
                    break
    images_array = np.array(images_array)
    true_classes = np.array(true_classes)

    print('Preprocessing in progress')
    print('FINISH')

    return images_array, true_classes, numbers_list

if __name__ == "__main__":
    # images_array, true_classes, numbers_list = get_imagenet()
    # save(images_array, 'imageNet_image')
    # save(true_classes, 'imageNet_label')
    # save(numbers_list, 'imageNet_numbers')

    images_array = load('imageNet_image')
    # images_array = np.transpose(images_array, (0, 3, 1, 2))
    val_transform =  transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    true_classes = load('imageNet_label')
    numbers_list = np.array(load('imageNet_numbers'))

    images_list = []
    for image in images_array:
        image = val_transform(image)
        images_list.append(image)
    dataset = TensorDataset(torch.stack(images_list), torch.from_numpy(numbers_list))
    data_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=1)
    
    # Classes label Dictionary
    class_mapping_dict, class_mapping_dict_number, mapping_class_to_number, mapping_number_to_class = get_mapping_class()

    # Use pretrained ResNet50
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    for param in resnet50.parameters():
        param.requires_grad = False
    resnet50.to(device)

    # Evaluate the performance of pretrained ResNet50.
    resnet50.eval()

    label_list = []
    with torch.no_grad():
         for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            image, label = data
            image = image.to(device)
            # Forward pass.
            outputs = resnet50(image.float())
            prediction = class_mapping_dict_number[outputs.cpu().data.numpy().argmax()]
            label_list.append(prediction)

    random_index = random.sample(range(0, 1000), 5)

    # Display True Label Image.
    plt.figure(figsize=(20, 20))
    plt.suptitle('True Label', x = 0.5, y = 0.6)
    for i in range(5):
        ax = plt.subplot(1, 5, i + 1)
        ax.imshow(images_array[random_index[i]])
        current_label = true_classes[random_index[i]]
        ax.set_title(f'{current_label}')

    plt.savefig("ImageNet_true_label.png")

        # Display True Label Image.
    plt.figure(figsize=(20, 20))
    plt.suptitle('Predict Label', x = 0.5, y = 0.6)
    for i in range(5):
        ax = plt.subplot(1, 5, i + 1)
        ax.imshow(images_array[random_index[i]])
        current_label = label_list[random_index[i]]
        ax.set_title(f'{current_label}')

    plt.savefig("ImageNet_predict_label.png")
