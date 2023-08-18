import sys
sys.path.append("../")
from simulation.simulator import Simulator
import torch
import os
from agents.MLP_agent import MLPagent
from infrastructure.utils import *
from infrastructure.gnm_utils import *
from infrastructure.training_utils import save_prediction_plots
from omegaconf import DictConfig, OmegaConf
import infrastructure.pytorch_util as ptu
from PIL import Image as PILImage
import hydra
import quaternion

TOPOMAP_IMAGES_DIR = "./data/topomap/images"
TOPOMAP_ACTIONS_DIR = "./data/topomap/actions"
VEL_THRESHOLD = 0.05
THETA_THRESHOLD = 0.10

def load_topomap_and_traj_and_init_state(params):
    # Load topo map
    topomap_filenames = sorted(os.listdir(os.path.join(
    TOPOMAP_IMAGES_DIR, params['topomap_dir'])), key=lambda x: int(x.split(".")[0]))
    topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{params['topomap_dir']}"
    num_nodes = len(os.listdir(topomap_dir))
    topomap = []
    for i in range(num_nodes):
        image_path = os.path.join(topomap_dir, topomap_filenames[i])
        topomap.append(PILImage.open(image_path))
    print("-"*50)
    print("num_nodes: ", num_nodes)
    print("-"*50)

    ref_traj = load(filepath=os.path.join(os.path.join(
    TOPOMAP_ACTIONS_DIR, params['topomap_dir']), params['topomap_dir'] + "_path.pkl"))
    init_state = load(filepath=os.path.join(os.path.join(
    TOPOMAP_ACTIONS_DIR, params['topomap_dir']), params['topomap_dir'] + "_init_state.pkl"))
    
    return topomap, ref_traj, init_state

def update_figure(topomap, observations, simulator, count_steps, demonstra_traj, vis_frames, curr_node):
    next_sg_img = topomap[curr_node]
    observations['depth'] = np.asarray(next_sg_img) # use as short goal image.
    info = simulator.env.get_metrics()
    set_agent_map_coord_and_angle(simulator.agent.state, info, simulator)  # Update position and rotation.
    
    frame = simulator.observations_to_image_with_policy(observations, info)
    cv2.putText(frame, f'steps: {count_steps}', (50, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    cv2.putText(frame, f'node: {curr_node}', (np.asarray(next_sg_img).shape[1]+70,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    cv2.putText(frame, 'curr obs', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(frame, 'short goal', (np.asarray(next_sg_img).shape[1]+70,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("Habitat", simulator.transform_rgb_bgr(frame))

    # Draw demonstration trajectory.
    maps.draw_path(
        top_down_map=info['top_down_map']['map'],
        path_points=demonstra_traj,
        color=10,
        thickness=10,
    )

    # Append the current frame into the frame series. 
    vis_frames.append(frame)

    return np.asarray(next_sg_img), observations['rgb']

def add_batch_dim(img:np.ndarray) -> np.ndarray:
    """
    Add a batch dimension to a given image and return it as a numpy array.

    Args:
    img: A numpy array representing an image with shape HxWxC.

    Returns:
    A numpy array with shape 1xCxHxW, where C is the number of channels,
    H is the height, and W is the width of the image.
    """
    transposed_img = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    return transposed_img

class PolicyEvaluator(object):
    def __init__(self, filepath) -> None:
        self.params = OmegaConf.load(os.path.join(filepath, '.hydra/config.yaml'))
        self.params['log_dir'] = filepath

        self.logs = load(os.path.join(filepath, 'experiment_result.pkl'))
        self.device = ptu.init_gpu(use_gpu=not self.params['no_gpu'], gpu_id=self.params['which_gpu'])

        # Load model.
        self.resnet50 = pretrained_ResNet50(self.device)
        self.agent = MLPagent(self.params)
        self.agent.load_model_params(os.path.join(filepath, 'MLP_policy.pt'))
        self.agent.model.to(self.device)

    def plot_metrics(self):
        """
        Save training loss and validation loss figure under the same folder as the policy.
        """
        log_metrics_figure(
                self.logs['train_loss'],
                self.logs['valid_loss'],
                path= os.path.join(self.params['log_dir'], 'metrics/'),
                loss=True
        )

    def plot_regression_figure(self):
        """
        Save True vel_x vs. Predicted vel_x and True steering_angle vs. Predicted steering_angle figure
        under the same folder as the policy.
        """ 
        # Make sure the regression plot use the same dataloader as the one for training.
        sub_dir = os.path.join(self.params['log_dir'], 'metrics/')
        train_loader, valid_loader = self.get_dataloader(preprocessed_features=True, preprocessed_dataset=True)
        save_prediction_plots(self.agent.model, valid_loader, self.device, val_data=True, path=sub_dir)
        save_prediction_plots(self.agent.model, train_loader, self.device, val_data=False, path=sub_dir)

    def evaluate(self, batch_name, log_figure: bool=True):
        """
        Evaluate policy on random images from training set.
        """
        dataset_dir = os.path.join(os.getcwd(), '../data/')
        a1_dataset = load(filepath=dataset_dir + 'a1_dataset.pkl')
        interval = self.params['interval']

        # Random sample 9 images and their commands.
        images, cmds = a1_dataset[batch_name]
        n = images.shape[0]
        sample_index  = np.random.randint(0, n, 9)
        while np.max(sample_index) + interval >= n:
            sample_index  = np.random.randint(0, n, 9)
        
        samples, goal_imgs = images[sample_index], images[sample_index + interval]
        samples_labels, goal_imgs_labels = cmds[sample_index], cmds[sample_index + interval] 
        actions = self.get_batch_actions((samples, samples_labels), (goal_imgs, goal_imgs_labels))

        # Plot raw images and goal images.
        if log_figure:
            log_dir = os.path.join(self.params['log_dir'], 'prediction/')
            self.log_images(samples, samples_labels, name=batch_name+'_raw_imgs', path=log_dir)
            self.log_images(goal_imgs, goal_imgs_labels, name=batch_name+'_goal_imgs', path=log_dir)
            self.log_images(samples, actions, name=batch_name+'_prediction', path=log_dir)

        print("images shape: ", samples.shape)
        print("action: ", actions)
        print('-'*100)

    def get_batch_actions(self, origin_data, goal_data, batch_size: int=3):
        """
        Return the command tuples (vel_x, steering angle) from the trained policy 
        input by the given original images numpy array and goal images numpy array. 
        """
        origin_imgs_features, origin_cmds = extract_batch_feature(
            self.device, self.resnet50, origin_data, batch_size=batch_size
        )

        goal_imgs_features, goal_cmds = extract_batch_feature(
            self.device, self.resnet50, goal_data, batch_size=batch_size
        )

        n = len(origin_imgs_features)
        origin_imgs_features = torch.tensor(origin_imgs_features).reshape((n, 2048)).to(self.device)
        goal_imgs_features = torch.tensor(goal_imgs_features).reshape((n, 2048)).to(self.device)
        combined_features = torch.cat((origin_imgs_features, goal_imgs_features), -1).to(self.device)

        # Concatenate images features.
        self.agent.model.eval()
        with torch.no_grad():
            action = self.agent.model(combined_features)

        return action.detach().cpu().numpy()

    def log_images(self, images, labels, name: None, path:str):
        # Create the directory if it doesn't exist.
        if not os.path.exists(path):
            os.makedirs(path)        

        rows = 3
        columns = 3
        fig = plt.figure(figsize=(10, 8))
        for i in range(9):
            fig.add_subplot(rows, columns, i+1)
            # Change into HxWxC.
            N, C, H, W = images.shape
            curr_img = images[i, :].reshape((H, W, C))
            plt.imshow(curr_img)
            plt.title(f"{labels[i]}")
        plt.suptitle(name)
        plt.savefig(os.path.join(path,  f'{name}.png'))

    def get_dataloader(self, preprocessed_features: bool, preprocessed_dataset: bool):
        """
        Load dataset.
        """
        dataset_dir = os.path.join(os.getcwd(), '../data/')
        if preprocessed_features:
            feature_dataset = load(filepath=dataset_dir + 'feature_dataset.pkl')
        else:
            # Load a1 dataset and extract features.
            if not preprocessed_dataset:
                a1_dataset = get_a1_dataset()
                save(a1_dataset, filepath=dataset_dir + 'a1_dataset.pkl')
            else:
                a1_dataset = load(filepath=dataset_dir + 'a1_dataset.pkl')
            
            resnet50 = pretrained_ResNet50(self.device)
            feature_dataset = {x: extract_batch_feature(self.device, resnet50, a1_dataset[x]) for x in ['part_1', 'part_2']}
            save(feature_dataset, filepath=dataset_dir + 'feature_dataset.pkl')

        # TODO: The order doesn't change in the a1_dataset.
        # Add plot option for cmds after averaging, I 
        # want to see if the cmds follows the similar trends
        # as before the averaging but less oscillating.

        # Noted that since I didn't manually set seed,
        # so each time I run the code without the preprocessed flag,
        # a1_dataloader will randomly reshuffle the result will vary 
        # as well.

        train_loader, valid_loader = get_a1_dataloader(
            feature_dataset, 
            batch_size=self.params['batch_size'], 
            interval=self.params['interval']
            )
        return train_loader, valid_loader

    def plot_commands(self):
        # Create the directory if it doesn't exist.
        path = os.path.join(self.params['log_dir'], 'metrics/')
        if not os.path.exists(path):
            os.makedirs(path)

        dataset_dir = os.path.join(os.getcwd(), '../data/')
        a1_dataset = load(filepath=dataset_dir + 'a1_dataset.pkl')

        cmds_part_1 = a1_dataset['part_1'][1]
        cmds_part_2 = a1_dataset['part_2'][1]
        vel_part_1, theta_part_1 = cmds_part_1[:, 0], cmds_part_1[:, 1]
        vel_part_2, theta_part_2 = cmds_part_2[:, 0], cmds_part_2[:, 1]

        plt.figure(figsize=(10, 7))
        plt.plot(
            vel_part_1, color='tab:blue', linestyle='-',
            label='forward_speed(vel_x) (part_1)'
        )

        plt.plot(
            vel_part_2, color='tab:red', linestyle='-',
            label='forward_speed(vel_x) (part_2)'
        )

        plt.xlabel('Time(t)')
        plt.ylabel('Speed(v)')
        plt.legend()
        plt.savefig(os.path.join(path, 'velocity.png'))

        plt.figure(figsize=(10, 7))
        plt.plot(
            theta_part_1, color='tab:blue', linestyle='-',
            label='steering_angle (part_1)'
        )

        plt.plot(
            theta_part_2, color='tab:red', linestyle='-',
            label='steering_angle (part_2)'
        )       

        plt.xlabel('Time(t)')
        plt.ylabel(r'Steering angle($\theta$)')
        plt.legend()
        plt.savefig(os.path.join(path, 'steering_angle.png'))
    
    def plot_simulation_actions(self):
        log_dir = self.params['log_dir']
        actions = load(os.path.join(log_dir, 'simulation/actions.npy'))

        plt.figure(figsize=(10, 7))
        plt.plot(
            actions[:, 0], color='tab:red', linestyle='-',
            label='forward_speed(vel_x)'
        )

        plt.xlabel('Time(t)')
        plt.ylabel('Speed(v)')
        plt.legend()
        plt.savefig(os.path.join(log_dir, 'simulation/vel_simulation.png'))


        plt.figure(figsize=(10, 7))
        plt.plot(
            actions[:, 1], color='tab:blue', linestyle='-',
            label='steering_angle'
        )

        plt.xlabel('Time(t)')
        plt.ylabel(r'Steering angle($\theta$)')
        plt.legend()
        plt.savefig(os.path.join(log_dir, 'simulation/steering_angle_simulation.png'))

    def run_sim(self, params:dict)->None:
        """
        run policy in habitat simulation.

        Args:
            params(dict) : The dictinary contains the mapping from command flag
                               to user response.
        Returns:
            None
        """
        # step 1: load short goal image and get the first observation.
        topomap, rollout_traj, init_state = load_topomap_and_traj_and_init_state(params)

        simulator = Simulator()
        simulator.init_env(params=None)
        observations = simulator.env.reset()
        info = simulator.env.get_metrics()
        observations = simulator.sim.get_observations_at(
            position=init_state[0],
            rotation=init_state[1],
            keep_agent_at_new_pose=True
        )
        set_agent_map_coord_and_angle(simulator.agent.state, info, simulator)
        draw_rollout_traj(rollout_traj, info, simulator)
        demonstra_traj = [get_agent_map_coord(simulator.agent.state.position, info, simulator)]
        vis_frames = []  # for logging video. 
        
        # step 2: write the loop code, should be similar with the loop code of GNM.
        count_steps = 0
        curr_node = 1
        node_counter = 0
        while not simulator.env.episode_over:
            next_sg_img, curr_obs_img = update_figure(topomap, observations, simulator, count_steps, demonstra_traj, vis_frames, curr_node)
            
            if params['mode'] == 'policy' or params['mode'] == 'video':
                keystroke = cv2.waitKey(20) if params['mode'] else cv2.waitKey(500)
                if keystroke == ord('f'): 
                    simulator.env.close()
                    return

                action = self.get_batch_actions(
                    (add_batch_dim(curr_obs_img), np.random.rand(1)),
                    (add_batch_dim(next_sg_img), np.random.rand(1)),
                    batch_size=1
                )

                # interpolate the next position
                agent_state = simulator.agent.state
                vel, yaw_angle = action[0][0], action[0][1]

                # TODO: change the rotation
                agent_rotation_vector = quaternion.as_rotation_vector(agent_state.rotation)
                agent_rotation_vector[1] += yaw_angle * 0.2
                next_rotation = quaternion.from_rotation_vector(agent_rotation_vector)
           
                next_position = np.array([
                    agent_state.position[0]-np.sin(agent_rotation_vector[1]) * vel * 0.4,
                    agent_state.position[1],
                    agent_state.position[2] -np.cos(agent_rotation_vector[1]) * vel * 0.4
                ])

                print("policy action: ", action)
                print("agent position: ", agent_state.position, "\n")

                # render image
                obs = simulator.sim.get_observations_at(
                    position=next_position,
                    rotation=next_rotation,
                    keep_agent_at_new_pose=True
                )

                # logging 
                observations['rgb'] = obs['rgb']
                demonstra_traj.append(get_agent_map_coord(simulator.agent.state.position, info, simulator))

                # update short goal node if the policy thinks it is close enough. 
                if vel < VEL_THRESHOLD and yaw_angle < THETA_THRESHOLD:
                    node_counter += 1
                    if node_counter == 3:
                        curr_node += 1
                        node_counter = 0

            if params['mode'] == 'manual':
                keystroke = cv2.waitKey(0)
                if chr(keystroke) in simulator.actions.keys():
                    action, log_action = simulator.actions[chr(keystroke)]
                    print(f"actions: {log_action}", "\n")
                else:
                    print("INVALID KEY", "\n")
                    continue
                observations = simulator.env.step(action)

            count_steps += 1

            if curr_node == len(topomap) or count_steps >= 120:
                simulator.env.close()
                print("Finish episode...")
                return

@hydra.main(version_base=None, config_path="../conf", config_name="policy_config")
def test(cfg):
    print(OmegaConf.to_yaml(cfg))
    print('-'*50)

    filepath = os.path.join(os.getcwd(), '../conf_outputs/2023-' + cfg['policy_dir'])
    evaluator = PolicyEvaluator(filepath)
    evaluator.run_sim(cfg)
    


# DO NOT USE THIS FUNCTION! NEED TO CHANGE FOR COUPLING WITH THE EXISTING PIPELINE!
def main(): 
    import argparse
    # TODO: same actions problem: highly likely due to the images improper reshape or transform.
    # The image shape of a1 image(3, 224, 240) is different than the shape of the simulation image(256, 256, 3).

    # TODO: Check what kind of operation you did for a1 raw image data. That maybe the things 
    # that causes different.

    # python3 evaluate_policy.py -s -d 
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', '-dir', default='04-17/16-47', required=False)
    parser.add_argument('--evaluate', '-e', action='store_true')
    parser.add_argument('--ep_name', '-n', type=str, default='ep_actions.pkl', required=False)

    # args for simulation.
    parser.add_argument('--sim', '-s', action='store_true')
    parser.add_argument('--no-policy', dest='policy', action='store_false')
    parser.add_argument("--display", '-d', dest="display", action="store_true")
    parser.add_argument("--log_action", '-a', action="store_true")
    parser.set_defaults(policy=True, display=False, log_action=False)
    args = parser.parse_args()
    params = vars(args)

    filepath = os.path.join(os.getcwd(), '../conf_outputs/2023-' + params['directory'])
    evaluator = PolicyEvaluator(filepath)

    if params['sim']:
        evaluator.run_sim(params)

    if params['evaluate']:
        evaluator.evaluate(batch_name='part_1', log_figure=True)
        evaluator.plot_commands()
        evaluator.plot_metrics()
        evaluator.plot_regression_figure()

if __name__ == "__main__":
    test()