import deeplake
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import torch

# Load Tiny ImageNet Dataset
ds_train = deeplake.load("hub://activeloop/tiny-imagenet-train")
val_transform =  transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

batch_size = 4
data_loader = ds_train.pytorch(
    num_workers = 0,  
    transforms= {'images': val_transform, 'labels': None},
    batch_size = batch_size, 
    decode_method = {'images': 'numpy'}
    )

# Use pretrained ResNet50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet50.to(device)


# Evaluate the performance of pretrained ResNet50.
resnet50.eval()

with torch.no_grad():
    for i, data in enumerate(data_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data['images']
        labels = torch.squeeze(data['labels'])

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = resnet50(inputs.float())


# save_dataset(images, name='tiny_imagenet')
# file_path = '/home/yuquand/ResNet18/data/tiny_imagenet.pkl'
# with open(file_path, 'rb') as f:
#     images = pickle.load(f)
# print(f"Successfully loaded {file_path}")


# fig, axes = plt.subplots(3, 3)

# # iterate over the images and plot each one in a separate subplot
# for i, ax in enumerate(axes.flat):
#      # make sure the current image index is within the bounds of the images list
#     if  i < len(images):
#         ax.imshow(images[i])
#         ax.set_title(f'{labels[i]}')
#     # turn off the axis labels and ticks for all subplots
#     ax.set(xticks=[], yticks=[], xlabel='', ylabel='')

# # adjust the spacing between subplots to make them look nicer
# fig.tight_layout()

# plt.savefig(os.path.join('../outputs', 'Tiny ImageNet.png'))

