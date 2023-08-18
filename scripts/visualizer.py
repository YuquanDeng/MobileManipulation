import os
from omegaconf import DictConfig, OmegaConf
import numpy as np
import matplotlib.pyplot as plt

class Visualizer(object):
    def __init__(self, filepath) -> None:
        self.params = OmegaConf.load(os.path.join(filepath, '.hydra/config.yaml'))
        self.params['log_dir'] = filepath

    def plot_goal_imgs(self):
        # load numpy array and plot only one of them under the same directoy. 
         
        # Load intermediate goal images.
        imgs_dir = os.path.join(self.params['log_dir'], "simulation")
        imgs_name = {}
        for filename in os.listdir(imgs_dir): # Noted that os.listdir is unordered.
            if filename.startswith("goal_img_") and filename.endswith('.npy'):
                num_steps = int(filename[9:-4])
                imgs_name[num_steps] = filename
        sorted_imgs_name_key = sorted(imgs_name.keys())

        # load images and plot the figures. 
        imgs = []
        for num_steps in sorted_imgs_name_key:
            file = os.path.join(self.params['log_dir'], "simulation/" + imgs_name[num_steps])
            imgs.append(np.load(file))
        
        self.plot_figures(np.array(imgs), sorted_imgs_name_key, os.path.join(self.params['log_dir'], "simulation"))

    # Give a list of images, save the figures.
    def plot_figures(self, images, sorted_num_steps, path):
        # Create the directory if it doesn't exist.
        if not os.path.exists(path):
            os.makedirs(path)        

        rows = 2
        columns = 7
        fig = plt.figure(figsize=(10, 8))
        for i in range(14):
            fig.add_subplot(rows, columns, i+1)
            curr_img = self.transform_rgb_bgr(images[i, :])
            plt.imshow(curr_img)
            plt.title(f"{sorted_num_steps[i]}")

        plt.suptitle("Goal Images")
        plt.savefig(os.path.join(path, 'goal_images_figure.png'))

    def transform_rgb_bgr(self, image):
        return image[:, :, [2, 1, 0]]

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', '-d', default='04-17/16-47', required=False)
    args = parser.parse_args()
    params = vars(args)

    filepath = os.path.join(os.getcwd(), '../conf_outputs/2023-' + params['directory'])
    visualizer = Visualizer(filepath)
    visualizer.plot_goal_imgs()


if __name__ == "__main__":
    main()