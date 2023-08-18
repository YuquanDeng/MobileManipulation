import unittest
import numpy as np
import sys
import torch
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
sys.path.append("../")
from infrastructure.utils import get_batch_dataset, get_a1_dataset


class TestUtils(unittest.TestCase):
    def test_get_a1_dataset(self):
        a1_dataset = get_a1_dataset()

        batch_1_images, batch_1_labels = a1_dataset['part_1']
        part1_last_cmds = np.array([[[1.1720, 0.1800],
                                    [1.2190, 0.2110],
                                    [1.1250, 0.0000],
                                    [0.8750, 0.0000],
                                    [0.7500, 0.0000],
                                    [0.0000, 0.0000],
                                    [0.0000, 0.0000],
                                    [0.0000, 0.0000],
                                    [0.0000, 0.0000],
                                    [0.0000, 0.0000]]])
        self.assertTrue((batch_1_labels[-10:, :] == part1_last_cmds).all())
        
        batch_2_images, batch_2_labels = a1_dataset['part_2']
        part2_last_cmds = np.array([ [0.3750, 0.0230],
                                    [0.4060, 0.0310],
                                    [0.4530, 0.0160],
                                    [0.4690, 0.0160],
                                    [0.3910, 0.0000],
                                    [0.3590, 0.0230],
                                    [0.4060, 0.0000],
                                    [0.4530, 0.0000],
                                    [0.4690, 0.0000],
                                    [0.5470, 0.0000]])
        self.assertTrue((batch_2_labels[-10:, :] == part2_last_cmds).all())
    
    def test_pretrained_ResNet50(self):
        #TODO: How to check if my pretrained model is reliable?

        classes_labels = {}
        with open('../data/imagenet_1000classes.txt', 'r') as file:
            # read all lines of the file into a list
            lines = file.readlines()

            for line in lines:
                # remove any whitespace at the beginning or end of the line
                line = line.strip()[:-1]
                if len(line) != 0:
                    index, label = line.split(":")[0],  line.split(":")[1]
                    classes_labels[int(index)] = label


        # Initialize pretrained model.
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in resnet50.parameters():
            param.requires_grad = False
        resnet50.to(device)
        resnet50.eval()
        print("Validation")

        # Load Validation data.
        a1_datasets = get_a1_dataset(normalized=False)

        val_transform =  transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Image index that we will display.
        img_index_list, label_list = [], []
        rows = 3
        columns = 3
        num_imgs = rows * columns
        for i in range(num_imgs):
            index = i * 100
            img_index_list.append(index) 
            curr_img = img_data[index].to(device).unsqueeze(0)
    
            # Evaluate Image.
            with torch.no_grad():
                prediction = resnet50(curr_img)
                label = classes_labels[prediction.cpu().data.numpy().argmax()]
                label_list.append(label)
        
        # Plot multiple images and corresponding classification results.
        fig = plt.figure(figsize=(10, 8))
        for i in range(num_imgs):
            fig.add_subplot(rows, columns, i+1)
            curr_img = torch.permute(img_data[img_index_list[i]], (1, 2, 0)).data.numpy()
            plt.imshow(curr_img)
            plt.title(f"{label_list[i]}")
        plt.savefig(os.path.join('../outputs',  'a1_data_resnet50_part1.png'))
    
    def test_get_a1_dataloader(self):
        feature_dataset = load_dataset(name='feature_dataset')
        batch_size = 1024

        for batch_name in feature_dataset:
            batch_feature, batch_label = feature_dataset[batch_name]

            print(f"batch name: {batch_name}")
            print(f"batch feature sizes: {batch_feature.size()}, batch label sizes: {batch_label.size()}")

            print(f"First 5 features: {batch_feature[:5, :]}\n")

        train_loader, val_loader = get_a1_dataloader(feature_dataset, batch_size)

        # # Fixed Feature Extractor
        # resnet50 = pretrained_ResNet50(device)
        # # Print module summary of the pretrained ResNet50
        # summary(resnet50, (3, 224, 224))

class TestGetBatchDatasetMethods(unittest.TestCase):
    def test_get_batch_dataset_return_type(self):
        # Check return types for loading data from part_1 directory.
        batch_1_dir = '/data/yuquand/a1_demo/part_1'
        batch_1_images, batch_1_labels = get_batch_dataset(batch_1_dir)

        self.assertIsInstance(batch_1_images, np.ndarray)
        self.assertIsInstance(batch_1_labels, np.ndarray)

        # Check return types for loading data from part_2 directory.
        batch_2_dir = '/data/yuquand/a1_demo/part_2'
        batch_2_images, batch_2_labels = get_batch_dataset(batch_2_dir)

        self.assertIsInstance(batch_2_images, np.ndarray)
        self.assertIsInstance(batch_2_labels, np.ndarray)

    def test_get_batch_dataset_first_ten_pairs(self):
        # Check the order of labels and manual check 
        # the pairs of images and labels.
        batch_1_dir = '/data/yuquand/a1_demo/part_1'
        batch_1_images, batch_1_labels = get_batch_dataset(batch_1_dir)

        batch_1_first_ten_cmds = np.array([[[0.0000, 0.0000],
                                            [0.0000, 0.0000],
                                            [0.0000, 0.0000],
                                            [0.0000, 0.0000],
                                            [0.0000, 0.0000],
                                            [0.0000, 0.0000],
                                            [0.0000, 0.0000],
                                            [0.0000, 0.0000],
                                            [0.0000, 0.0000],
                                            [0.0000, 0.0000]] ])

        self.assertTrue((batch_1_labels[:10, :] == batch_1_first_ten_cmds).all())


        batch_2_dir = '/data/yuquand/a1_demo/part_2'
        batch_2_images, batch_2_labels = get_batch_dataset(batch_2_dir)
        batch_2_first_ten_cmds = np.array([ [0.250, 0.0000],
                                            [0.250, 0.0000],
                                            [0.266, 0.0000],
                                            [0.266, 0.0000],
                                            [0.266, 0.0000],
                                            [0.266, 0.0000],
                                            [0.266, 0.0000],
                                            [0.266, 0.0000],
                                            [0.297, 0.0000],
                                            [0.281, 0.0000] ])
        
        self.assertTrue((batch_2_labels[:10, :] == batch_2_first_ten_cmds).all())

    def test_get_batch_dataset_middle_ten_pairs(self):
        # Check pairs of images from 01167 to 01176.
        batch_1_dir = '/data/yuquand/a1_demo/part_1'
        batch_1_images, batch_1_labels = get_batch_dataset(batch_1_dir)

        batch_1_middle_ten_cmds = np.array([[[0.938, 0.086],
                                            [1.016, 0.164],
                                            [1.000, 0.180],
                                            [1.047, 0.164],
                                            [1.031, 0.086],
                                            [0.828, 0.031],
                                            [0.875, 0.039],
                                            [0.938, 0.070],
                                            [1.016, 0.078],
                                            [0.969, 0.117]] ])
        
        self.assertTrue((batch_1_labels[1167:1177, :] == batch_1_middle_ten_cmds).all())

        # Check for part_2. 
        batch_2_dir = '/data/yuquand/a1_demo/part_2'
        batch_2_images, batch_2_labels = get_batch_dataset(batch_2_dir)
        batch_2_middle_ten_cmds = np.array([ [  [0.000, 0.000],
                                                [0.000, 0.000],
                                                [0.078, 0.000],
                                                [0.188, 0.000],
                                                [0.594, 0.000],
                                                [0.438, 0.000],
                                                [0.422, 0.000],
                                                [0.391, 0.023],
                                                [0.375, 0.023],
                                                [0.562, 0.062]  ] ])

        self.assertTrue((batch_2_labels[1167:1177, :] == batch_2_middle_ten_cmds).all())

    def test_get_batch_dataset_last_ten_pairs(self):
        # Check the order of labels and manual check 
        # the pairs of images and labels.
        batch_1_dir = '/data/yuquand/a1_demo/part_1'
        batch_1_images, batch_1_labels = get_batch_dataset(batch_1_dir)

        batch_1_last_ten_cmds = np.array([ [[1.1720, 0.1800],
                                            [1.2190, 0.2110],
                                            [1.1250, 0.0000],
                                            [0.8750, 0.0000],
                                            [0.7500, 0.0000],
                                            [0.0000, 0.0000],
                                            [0.0000, 0.0000],
                                            [0.0000, 0.0000],
                                            [0.0000, 0.0000],
                                            [0.0000, 0.0000]] ])

        self.assertTrue((batch_1_labels[-10:, :] == batch_1_last_ten_cmds).all())


        batch_2_dir = '/data/yuquand/a1_demo/part_2'
        batch_2_images, batch_2_labels = get_batch_dataset(batch_2_dir)
        batch_2_last_ten_cmds = np.array([  [0.3750, 0.0230],
                                            [0.4060, 0.0310],
                                            [0.4530, 0.0160],
                                            [0.4690, 0.0160],
                                            [0.3910, 0.0000],
                                            [0.3590, 0.0230],
                                            [0.4060, 0.0000],
                                            [0.4530, 0.0000],
                                            [0.4690, 0.0000],
                                            [0.5470, 0.0000] ])
        
        self.assertTrue((batch_2_labels[-10:, :] == batch_2_last_ten_cmds).all())


if __name__ == '__main__':
    unittest.main()

    # python3 -m unittest test_utils.TestUtils.test_get_a1_dataset
    # python3 -m unittest test_utils.TestGetBatchDatasetMethods.test_get_batch_dataset_return_type
    # python3 -m unittest test_utils.TestGetBatchDatasetMethods.test_get_batch_dataset_last_ten_pairs
    # python3 -m unittest test_utils.TestGetBatchDatasetMethods.test_get_batch_dataset_first_ten_pairs