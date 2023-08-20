import habitat
import habitat_sim
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import copy
from PIL import Image
import torchvision.transforms as transforms
import magnum as mn
import yaml
from habitat.utils.visualizations import maps
from typing import cast, Dict, List
import quaternion



from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import HabitatSimRGBSensorConfig

# TOPDOWN MAP
from habitat.core.agent import Agent
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
    tile_images,
    draw_collision,
)
import hydra
from omegaconf import DictConfig, OmegaConf

# Utils.
sys.path.append("../")
from utilities.utils import save
from utilities.gnm_utils import set_agent_map_coord_and_angle 

ROBOT_CONFIG_PATH = os.path.join(os.getcwd(), "../configs/habitat_agent.yaml")
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]


UP_ROTATION = quaternion.quaternion(np.quaternion(-0.714877784252167, 0, 0.699249446392059, 0))
LEFT_ROTATION = quaternion.quaternion(np.quaternion(-0.99993896484375, 0, -0.011050783097744, 0))
DOWN_ROTATION = quaternion.quaternion(np.quaternion(-0.699249565601349, 0, -0.714877665042877, 0))
RIGHT_ROTATION = quaternion.quaternion(np.quaternion(-0.0110505819320679, 0, -0.999939024448395, 0))

BOTTOM_LEFT_POS = np.array([-5.062964, 0.11702011, 9.055218])
BOTTOM_RIGHT_POS = np.array([-4.0921426, 0.28621453, 19.0362])
UPPER_LEFT_POS = np.array([2.9476223, 0.1279917, 9.445251])
UPPER_RIGHT_POS = np.array([2.4062703, 0.12521705, 18.892534])

INIT_POSITION = BOTTOM_LEFT_POS
INIT_ROTATION = UP_ROTATION

POINT_NAV_POSITION = np.array([-7.3699493,   0.08276175,  6.5762997])
POINT_NAV_ROTATION = quaternion.quaternion(np.quaternion(0.56448882818222, 0, 0.825440764427185, 0))

class Simulator(object):
    def __init__(self) -> None:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        self.config = habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")
        

        self.actions = {
            "w": (HabitatSimActions.move_forward, 'FORWARD'), 
            "a": (HabitatSimActions.turn_left, 'LEFT'),
            "d": (HabitatSimActions.turn_right, 'RIGHT'), 
            "f": (HabitatSimActions.stop, 'STOP'),
            "SAVE_KEY": "s"
        }

        self.threshold = {
            "vel_x": 0.05,
            "theta": 0.05,
            'forward': 0.12,
            'left': 0.12
        }

    def init_env(self, params=None):
        with habitat.config.read_write(self.config):
            if params != None:
                agent_config = get_agent_config(sim_config=self.config.habitat.simulator)
                width = params['resolution']
                height = params['resolution']

                if params['width'] != params['height']:
                    width = params['width']
                    height = params['height']

                agent_config.sim_sensors.update(
                    {"rgb_sensor": HabitatSimRGBSensorConfig(
                        width=width,
                        height=height,
                        position=params['sensor_position'],
                        orientation=params['sensor_orientation'],
                        hfov=params['hfov']
                    )}
                )

            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=False,
                        draw_border=True,
                        draw_shortest_path=False,
                        draw_view_points=True,
                        draw_goal_positions=False,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
        )
        self.env = habitat.Env(config=self.config)
        self.sim = self.env.sim
        self.agent = self.sim.agents[0]

        keys = list(self.agent.agent_config.action_space.keys())

        print("action space: ", self.agent.agent_config.action_space[keys[1]].actuation.amount)
        print("-"*50)
    
    def recursive_iter(self, cfg):
        """
        Log omegaconf.DictConfig object key values pairs recursively on the terminal.
        """
        for key, value in cfg.items():
            if isinstance(value, (dict, DictConfig)):
                print(f"Entering {key}")
                self.recursive_iter(value)
            else:
                print(f"{key}: {value}")

    def add_batch_dim(self, img:np.ndarray) -> np.ndarray:
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

    def load_goal_imgs(self, log_dir):
        """
        Return an ordered dictionary with {num_steps: goal_img} pairs and the final goal image. 
        """
         # Load intermediate goal images.
        imgs_dir = os.path.join(log_dir, "simulation")
        imgs_name = {}
        for filename in os.listdir(imgs_dir): # Noted that os.listdir is unordered.
            if filename.startswith("goal_img_") and filename.endswith('.npy'):
                num_steps = int(filename[9:-4])
                imgs_name[num_steps] = filename
        sorted_imgs_name_key = sorted(imgs_name.keys())
        sorted_imgs = {}
        for key in sorted_imgs_name_key:
            file = os.path.join(log_dir, "simulation/" + imgs_name[key])
            sorted_imgs[key] = self.transform_rgb_bgr(np.load(file))

        return copy.deepcopy(sorted_imgs)

    def get_actions(self, policy_evaluator, curr_obs_img, next_sg_img):
        action = policy_evaluator.get_batch_actions(
            (self.add_batch_dim(curr_obs_img), np.random.rand(1)),
            (self.add_batch_dim(next_sg_img), np.random.rand(1)),
            batch_size=1
        )

        # final_action = policy_evaluator.get_batch_actions(
        #     (self.add_batch_dim(curr_img), np.random.rand(1)),
        #     (self.add_batch_dim(final_img), np.random.rand(1)),
        #     batch_size=1
        # )
        return action

    def get_next_img(self, goal_imgs, count_steps):
        """
        Given a ordered list of goal images, find the next goal images 
        by the given count_steps. Always return the last goal image
        if the count_steps is greater than all steps in the list.

        goals_imgs would not be an empty dictionary. 
        @returns: image with next minimum steps > count_steps.
        """
        next_imgs_key = -1
        last_img_key = -1
        for num_steps in goal_imgs:
            last_img_key = num_steps

            if num_steps > count_steps:
                next_imgs_key = num_steps
                break
        # Either the last image key in the iteration or the key of the next greater integer.
        next_imgs_key  = last_img_key if next_imgs_key == -1 else next_imgs_key
        
        return goal_imgs[next_imgs_key], next_imgs_key
    
    def cont2discrete_action(self, cont_action):
        """
        Transform a1 actions (vel_x, theta) into simulation discrete action
        (left, right, forward, finish).
        """
        vel_x , theta = cont_action[0][0], cont_action[0][1]
        action_keys = ['forward', 'left', 'right']

        # observation 1: Both of the action will drop under 0.05 if the policy thinks
        #                the current image similar to the goal image. The problem is 
        #                those goal images are intermediate goal image. 
        # One potential issue: maybe the agent takes more steps that it thought.

        # If the next_img is not the last goal image, move forward.
        # Terminate otherwise.

        # goal_imgs, curr_img index   
        if vel_x < self.threshold['vel_x'] and theta < self.threshold['theta']:

            return self.actions['finish']

        vel2action = self.actions['forward'] if vel_x > self.threshold['forward'] else None
        theta2action = self.actions['left'] if theta > self.threshold['left'] else None
        
        if vel2action is None and theta2action is None:
            return self.actions['left']
            return self.actions[action_keys[np.random.randint(0, 3)]] # Sample a random action from {left,  right, forward}
        elif vel2action is None:
            return theta2action
        elif theta2action is None:
            return vel2action
        else:
            return self.actions[action_keys[np.random.randint(0, 2)]] # Sample a random action from {left, forward}

    def transform_rgb_bgr(self, image):
        return image[:, :, [2, 1, 0]]

    def get_agent_action(self, display:bool):
        if display:
                keystroke = cv2.waitKey(0)
        else:
            keystroke = input("'a', 'w', 'd', or 'f' to quit: ")
            if len(keystroke) != 1:
                return None
            keystroke = ord(keystroke)
            
        # Invariant: keystroke is an integer representing 
        # the ASCII value of the key.
        if chr(keystroke) in self.actions.keys():
            action, log_action = self.actions[chr(keystroke)]
            print(f"actions: {log_action}")
            return action
        else:
            print("INVALID KEY")
            return None

    def start(self)->np.ndarray:
        """
        Initialize the simulation environment and return the first RGB observation.

        Args:
        display: A boolean indicating whether to display the RGB observation.

        Returns:
        A numpy array representing the first RGB observation of the environment.

        """
        print("Environment creation successful")
        observations = self.env.reset()
        print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
            observations["pointgoal_with_gps_compass"][0],
            observations["pointgoal_with_gps_compass"][1]))
        print("Agent stepping around inside environment.")
        return observations["rgb"]

    def terminate(self, action, observations)->None:
        """
        Terminate the simulation and print the success or failure of the navigation.

        Args:
        action: The last action taken by the agent.
        observations: A dictionary containing the last observation of the environment.

        """
        if (
            action == HabitatSimActions.stop
            and observations["pointgoal_with_gps_compass"][0] < 0.2
        ):
            print("you successfully navigated to destination point")
        else:
            print("your navigation was unsuccessful")

    def playback(self, actions, topomap_name:str, images_dir:str, actions_dir:str, init_pos=None, init_rotation=None, params=None) -> None:
        """
        Play the simulation with the given actions buffer and save the observation
        on each actions under the given topomap_name_dir.
        """
        # Create directory if not exists.

        # Initialize simulation environment.
        self.env.close()
        self.init_env(params=params)
        obs_img = self.start()

        if init_pos is not None and init_rotation is not None:
            observations = self.sim.get_observations_at(
                position=init_pos,
                rotation=init_rotation,
                keep_agent_at_new_pose=True
            )
            obs_img = observations['rgb']

        # path for top-down map measurement.
        path = []
        path.append(self.agent.state.position)
         
        # if you want to resize the image into the input size of GNM, do ...resize((85, 64)).
        Image.fromarray(obs_img).save(os.path.join(images_dir, "0.png"))

        for i in range(len(actions)):
            action = actions[i]
            observation = self.env.step(action)
            path.append(self.agent.state.position)
            Image.fromarray(observation['rgb']).save(os.path.join(images_dir, f"{i+1}.png"))

        save(file=path, filepath=os.path.join(actions_dir, topomap_name+'_path.pkl'))
        save(file=(init_pos, init_rotation), filepath=os.path.join(actions_dir, topomap_name+'_init_state.pkl'))
        print(f"Saved topomap images under {images_dir}")

    def set_camera_pose(self, agent_id, actions):
        agent = self.sim.agents[agent_id]
        agent.scene_node.translation += mn.Vector3(0, 0, actions[0][0])
        agent.scene_node.rotation = mn.Quaternion(mn.Vector3((0, 1.34, 0)), actions[0][1]).normalized()

    def run_sim(self, display:bool, log_action:bool=False) -> None: 
        # configs
        dir_path = os.path.join(os.getcwd(), './')
        filename = os.path.join(dir_path, 'real_time_obs.png')
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        # start the environment.
        obs_img = self.start()
        if display:
            cv2.imshow("RGB", self.transform_rgb_bgr(obs_img))
        else:
            Image.fromarray(obs_img).save(filename)
    
        # Simulation loop with command actions.
        ep_actions = []
        count_steps = 0
        while not self.env.episode_over:
            if display:
                keystroke = cv2.waitKey(0)
            else:
                keystroke = input("'a', 'w', 'd', or 'f' to quit: ")
                if len(keystroke) != 1:
                    continue
                keystroke = ord(keystroke)
                
            # Invariant: keystroke is an integer representing 
            # the ASCII value of the key.
            if chr(keystroke) in self.actions.keys():
                action, log_action = self.actions[chr(keystroke)]
                print(f"actions: {log_action}")
            else:
                print("INVALID KEY")
                continue 

            observations = self.env.step(action)
            obs_img = observations["rgb"]

            if display:
                cv2.imshow("RGB", self.transform_rgb_bgr(obs_img))
            else:
                Image.fromarray(obs_img).save(filename)
            count_steps += 1

            print("position: ", self.agent.scene_node.translation)
            print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
                observations["pointgoal_with_gps_compass"][0],
                observations["pointgoal_with_gps_compass"][1]))
            
            # Add actions.
            ep_actions.append(action)

        print("Episode finished after {} steps.".format(count_steps))
        # Delete the temporary display image.
        if os.path.exists(filename):
            os.remove(filename)
        self.terminate(action, observations)

        # Logging session.
        if log_action:
            save(file=ep_actions, filepath=os.path.join(os.getcwd(), './data/actions/ep_actions.pkl'))

    def set_initial_state(self, position, orientation):
        """
        Reset the initial position of an agent by the given position and orientation
        after running the simulation.
        """
        raise NotImplementedError

    def render(self, agent_id):
        obs = self.sim.get_sensor_observations(agent_id)
        obs_img = obs['rgb'][:, :, :3]
        return obs_img
    
    def run_sim_with_policy(self, policy_evaluator, display:bool, log_action:bool=False) -> None:
        # configs
        log_dir = policy_evaluator.params['log_dir']
        dir_path = os.path.join(os.getcwd(), './')
        filename = os.path.join(dir_path, 'real_time_obs.png')
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        
        # initialize env.
        sg_imgs = self.load_goal_imgs(log_dir)
        curr_obs_img = self.start()
        next_sg_img, sg_img_index = self.get_next_img(sg_imgs, count_steps=0)

        # display
        height = max(next_sg_img.shape[0], curr_obs_img.shape[0])
        width = next_sg_img.shape[1] + curr_obs_img.shape[1] + 20
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Navigation loop.
        count_steps = 0
        while not self.env.episode_over:
            if display:
                keystroke = cv2.waitKey(0)
            else:
                keystroke = input("enter ' ' for continue or 'f' for quit: ")
                if len(keystroke) != 1:
                    continue
                keystroke = ord(keystroke)

            if keystroke == ord('f'):
                self.env.step(self.actions[chr(keystroke)][0])
            
            # Get next action, set position, and render image.
            # action -> normalized(how?) -> denormalized -> set agent position and rotation.
            next_sg_img, sg_img_index = self.get_next_img(sg_imgs, count_steps)
            actions = self.get_actions(
                policy_evaluator, 
                curr_obs_img,
                next_sg_img
            )

            print(f"actions: {actions}")
            self.set_camera_pose(agent_id=0, actions=actions)
            curr_obs_img = self.render(agent_id=0)
            self.display_obs(canvas, curr_obs_img, next_sg_img, display, filename, count_steps, sg_img_index)
            count_steps += 1

        print("Episode finished after {} steps.".format(count_steps))
        # Delete the temporary display image.
        if os.path.exists(filename):
            os.remove(filename)
    
    def run_sim_with_topdown_map(self, log_action:bool=False, init_pos=None, init_rotation=None):
        # Initialize the env.
        observations = self.env.reset()
        info = self.env.get_metrics()
        if init_pos is not None and init_rotation is not None:
            observations = self.sim.get_observations_at(
                position=init_pos,
                rotation=init_rotation,
                keep_agent_at_new_pose=True
            )
            set_agent_map_coord_and_angle(self.agent.state, info, self)
        frame = self.observations_to_image(observations, info)
        
        info.pop("top_down_map")
        frame = overlay_frame(frame, info) 
        cv2.imshow("RGB", self.transform_rgb_bgr(frame))
        

        # Simulation loop with command actions.
        count_steps = 0
        ep_actions = []
        while not self.env.episode_over:
            keystroke = cv2.waitKey(0)
            
            # Invariant: keystroke is an integer representing 
            # the ASCII value of the key.
            if chr(keystroke) in self.actions.keys():
                action, log_action = self.actions[chr(keystroke)]
                print(f"actions: {log_action}")
            else:
                print("INVALID KEY")
                continue 

            observations = self.env.step(action)
            
            ep_actions.append(action)
            info = self.env.get_metrics()
            frame = self.observations_to_image(observations, info)

            info.pop("top_down_map")
            frame = overlay_frame(frame, info) 
            cv2.imshow("RGB", self.transform_rgb_bgr(frame))
            count_steps += 1
            
            
            print("agent position: ", self.agent.state.position)
            print("agent rotation: ", self.agent.state.rotation, '\n')

            # # TODO: comment it out if you want to see this.
            # print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
            #     observations["pointgoal_with_gps_compass"][0],
            #     observations["pointgoal_with_gps_compass"][1]))
            

        print("Episode finished after {} steps.".format(count_steps))
        self.terminate(action, observations)
        
        if log_action:
            return ep_actions
    
    def observations_to_image(self, observation: Dict, info: Dict) -> np.ndarray:
        r"""Generate image of single frame from observation and info
        returned from a single environment step().

        Args:
            observation: observation returned from an environment step().
            info: info returned from an environment step().

        Returns:
            generated image of a single frame.
        """
        render_obs_images: List[np.ndarray] = []
        for sensor_name in observation:
            if len(observation[sensor_name].shape) > 1:
                obs_k = observation[sensor_name]
                if not isinstance(obs_k, np.ndarray):
                    obs_k = obs_k.cpu().numpy()
                if obs_k.dtype != np.uint8:
                    obs_k = obs_k * 255.0
                    obs_k = obs_k.astype(np.uint8)
                if obs_k.shape[2] == 1:
                    obs_k = np.concatenate([obs_k for _ in range(3)], axis=2)
                render_obs_images.append(obs_k)

        assert (
            len(render_obs_images) > 0
        ), "Expected at least one visual sensor enabled."

        shapes_are_equal = len(set(x.shape for x in render_obs_images)) == 1
        if not shapes_are_equal:
            render_frame = tile_images(render_obs_images)
        else:
            render_frame = np.concatenate(render_obs_images, axis=1)

        # draw collision
        collisions_key = "collisions"
        if collisions_key in info and info[collisions_key]["is_collision"]:
            render_frame = draw_collision(render_frame)
    
        top_down_map_key = "top_down_map"
        if top_down_map_key in info:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(
                info[top_down_map_key], render_frame.shape[0]
            )
            render_frame = np.concatenate((render_frame, top_down_map), axis=1)
        return render_frame

    def display_obs(
            self, 
            canvas, 
            curr_obs_img:np.ndarray, 
            next_sg_img:np.ndarray, 
            display:bool, 
            filename:str,
            count_steps:int,
            sg_img_index:int
    ):
        if display:
            obs_h, obs_w = curr_obs_img.shape[0], curr_obs_img.shape[1]
            sg_h, sg_w = next_sg_img.shape[0], next_sg_img.shape[1]
            canvas[:obs_h, :obs_w] = self.transform_rgb_bgr(curr_obs_img)
            canvas[:sg_h, sg_w+20:sg_w+20+sg_w] = self.transform_rgb_bgr(next_sg_img)
            
            cv2.putText(canvas, f'steps: {count_steps}', (50, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            cv2.putText(canvas, f'node: {sg_img_index}', (next_sg_img.shape[1]+70,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            cv2.putText(canvas, 'curr obs', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.putText(canvas, 'short goal', (next_sg_img.shape[1]+70,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("RGB", canvas)
        else:
            Image.fromarray(curr_obs_img).save(filename)

    def observations_to_image_with_policy(self, observation: Dict, info: Dict) -> np.ndarray:
        # idea: use depth sensor bucket as placeholder for the short goal image.
        render_obs_images: List[np.ndarray] = []
        for sensor_name in observation:
            if len(observation[sensor_name].shape) > 1:
                obs_k = observation[sensor_name]
                if not isinstance(obs_k, np.ndarray):
                    obs_k = obs_k.cpu().numpy()
                if obs_k.dtype != np.uint8:
                    obs_k = obs_k * 255.0
                    obs_k = obs_k.astype(np.uint8)
                if obs_k.shape[2] == 1:
                    obs_k = np.concatenate([obs_k for _ in range(3)], axis=2)
                render_obs_images.append(obs_k)

        assert (
            len(render_obs_images) > 0
        ), "Expected at least one visual sensor enabled."

        shapes_are_equal = len(set(x.shape for x in render_obs_images)) == 1
        if not shapes_are_equal:
            render_frame = tile_images(render_obs_images)
        else:
            render_frame = np.concatenate(render_obs_images, axis=1)

        # draw collision
        collisions_key = "collisions"
        if collisions_key in info and info[collisions_key]["is_collision"]:
            render_frame = draw_collision(render_frame)
    
        top_down_map_key = "top_down_map"
        if top_down_map_key in info:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(
                info[top_down_map_key], render_frame.shape[0]
            )
            render_frame = np.concatenate((render_frame, top_down_map), axis=1)
        return render_frame

def main():
    """
    Test run_sim_with_topdown_map method
    """
    simulator = Simulator()
    simulator.init_env()
    # simulator.run_sim_with_topdown_map(
    #     log_action=False,
    #     init_pos=None,
    #     init_rotation=None,
    #     )


if __name__ == '__main__':
    main()
