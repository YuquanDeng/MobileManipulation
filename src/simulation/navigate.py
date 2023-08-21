# UTILS
import sys
sys.path.append("../")
from utilities.gnm_utils import *
from utilities.utils import load
from simulation.simulator import Simulator
from PIL import Image
import cv2
import torch
import os
from PIL import Image as PILImage
import yaml
import quaternion

# LOGGING & CONFIG 
from habitat.utils.visualizations.utils import (
    images_to_video
)
from omegaconf import DictConfig, OmegaConf
import hydra

TOPOMAP_DIR = "../../data/habitat/topomap/"
MODEL_CONFIG_PATH = os.path.join(os.getcwd(), "../../third_party/drive-any-robot/deployment/config/models.yaml")
MODEL_WEIGHTS_PATH = os.path.join(os.getcwd(), "../../third_party/drive-any-robot/deployment/model_weights")
ROBOT_CONFIG_PATH ="../configs/habitat_agent.yaml"

with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]


# config for agent position
INIT_POSITION = np.array([-6.0903454, 0.04104979, 7.8276253 ])
INIT_ROTATION = quaternion.quaternion(np.quaternion(-0.773101031780243, 0, 0.634282827377319, 0))
LINE_GOAL_POSITION = np.array([2.9814022, 0.0486111, 6.0204554])

# DEFAULT MODEL PARAMETERS (can be overwritten by model.yaml)
model_params = {
    "path": "large_gnm.pth", # path of the model in ../model_weights
    "model_type": "gnm", # gnm (conditioned), stacked, or siamese
    "context": 5, # number of images to use as context
    "len_traj_pred": 5, # number of waypoints to predict
    "normalize": True, # bool to determine whether or not normalize images
    "image_size": [85, 64], # (width, height)
    "normalize": True, # bool to determine whether or not normalize the waypoints
    "learn_angle": True, # bool to determine whether or not to learn/predict heading of the robot
    "obs_encoding_size": 1024, # size of the encoding of the observation [only used by gnm and siamese]
    "goal_encoding_size": 1024, # size of the encoding of the goal [only used by gnm and siamese]
    "obsgoal_encoding_size": 2048, # size of the encoding of the observation and goal [only used by stacked model]
}

# Load the model (locobot uses a NUC, so we can't use a GPU)
device = torch.device("cpu")

def load_topomap_and_traj_and_init_state(params):
    # Load topo map
    topomap_img_path =  os.path.join(os.getcwd(), TOPOMAP_DIR + '/' + params['dir']+'/images')
    topomap_filenames = sorted(os.listdir(topomap_img_path), key=lambda x: int(x.split(".")[0]))
    topomap_dir = f"{TOPOMAP_DIR}/{params['dir']}/images"
    num_nodes = len(os.listdir(topomap_dir))
    topomap = []
    for i in range(num_nodes):
        image_path = os.path.join(topomap_dir, topomap_filenames[i])
        topomap.append(PILImage.open(image_path))
    print("-"*50)
    print("number of nodes: ", num_nodes)

    topomap_action_path = os.path.join(os.getcwd(), TOPOMAP_DIR + '/' + params['dir'] + '/action')
    ref_traj = load(filepath=os.path.join(topomap_action_path, params['dir'] + "_path.pkl"))
    init_state = load(filepath=os.path.join(topomap_action_path, params['dir'] + "_init_state.pkl"))
    return topomap, ref_traj, init_state

def update_figure(topomap, closest_node, start, observations, simulator, count_steps, demonstra_traj, vis_frames):
        # display top down map with current observation and short goal image.
        next_sg_img = np.asarray(topomap[closest_node+start])
        observations['depth'] = np.asarray(next_sg_img) # use as short goal image.
        info = simulator.env.get_metrics()
        set_agent_map_coord_and_angle(simulator.agent.state, info, simulator)  # Update position and rotation.

        # TODO: delete this!  
        # cv2.imwrite('short_goal_img.jpg', simulator.transform_rgb_bgr(next_sg_img))
        # cv2.imwrite('curr_obs_img.jpg', simulator.transform_rgb_bgr(observations['rgb']))

        frame = simulator.observations_to_image_with_policy(observations, info)
        cv2.putText(frame, f'steps: {count_steps}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(frame, f'node: {closest_node+start}', (next_sg_img.shape[1]+10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(frame, 'curr obs', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
        cv2.putText(frame, 'short goal', (next_sg_img.shape[1]+10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
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

def local2worldCoordinate(position, rotation, waypoint):
    """
    Convert a relative waypoint represented in world coordinate.
    """
    rotation_matrix = quaternion.as_rotation_matrix(rotation)
    T_world_camera = np.eye(4)
    T_world_camera[0:3, 0:3] = rotation_matrix
    T_world_camera[0:3, 3] = position
    waypoint_w = T_world_camera @ waypoint

    return waypoint_w

def get_model_and_model_params(params):
    """
    Load model parameters and the pretrained GNM model. 
    """
    # Load model parameters
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_config = yaml.safe_load(f)
    for param in model_config:
        model_params[param] = model_config[param]

    # Load model weight
    model_filename = model_config[params['model']]["path"]
    model_path = os.path.join(MODEL_WEIGHTS_PATH, model_filename)
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    model = load_model(
        model_path,
        model_params["model_type"],
        model_params["context"],
        model_params["len_traj_pred"],
        model_params["learn_angle"], 
        model_params["obs_encoding_size"], 
        model_params["goal_encoding_size"],
        model_params["obsgoal_encoding_size"],
        device,
    )
    model.eval()

    return model, model_params

def run_simulation_loop(params, model, model_params, topomap, rollout_traj, init_state):
    # Habitat env
    simulator = Simulator()
    simulator.init_env(params=params)
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
    

    # TODO: Can try random crop or feed different images.
    context_queue = []
    for _ in range(model_params["context"] + 1):
        context_queue.append(PILImage.fromarray(observations['rgb']))

    # Node Initialization.
    closest_node = 0
    assert -1 <= params['goal_node'] < len(topomap), "Invalid goal index"
    if params['goal_node'] == -1:
        goal_node = len(topomap) - 1
    else:
        goal_node = params['goal_node']
    reached_goal = False

    # Navigation loop
    count_steps = 0
    while not simulator.env.episode_over:
        # invariant: len(context_queue) = model_params["context"]+1,
        # so you can safely delete the if condition after checking that.
        if len(context_queue) > model_params["context"]:
            start = max(closest_node - params['radius'], 0)
            end = min(closest_node + params['radius'] + 1, goal_node)
            distances = []
            waypoints = []
            for sg_img in topomap[start: end + 1]:
                transf_obs_img = transform_images(context_queue, model_params["image_size"])
                transf_sg_img = transform_images(sg_img, model_params["image_size"])
                dist, waypoint = model(transf_obs_img, transf_sg_img) 
                distances.append(to_numpy(dist[0]))
                waypoints.append(to_numpy(waypoint[0]))

            # look for closest node
            closest_node = np.argmin(distances)
            # chose subgoal and output waypoints
            if distances[closest_node] > params['close_threshold']:
                chosen_waypoint = waypoints[closest_node][params['waypoint']]
            else:
                chosen_waypoint = waypoints[min(
                    closest_node + 1, len(waypoints) - 1)][params['waypoint']]

            update_figure(topomap, closest_node, start, observations, simulator, count_steps, demonstra_traj, vis_frames)
 
            if params['mode'] == 'policy' or params['mode'] == 'video':
                keystroke = cv2.waitKey(1) if params['mode'] == 'policy' else cv2.waitKey(1)
                if keystroke == ord('f'): 
                    simulator.env.close()
                    return

                # normalized the waypoint and match habitat coordinate system. 
                normalized_waypoint = np.copy(chosen_waypoint)
                normalized_waypoint[:2] *= MAX_V
                normalized_pt_x, normalized_pt_y, pt_cos, pt_sin = tuple(normalized_waypoint)
                normalized_xzy = np.array([-normalized_pt_y, 0, -normalized_pt_x, 1])

                # interpolate position
                agent_state = simulator.agent.state
                normalized_xzy_in_w = local2worldCoordinate(agent_state.position, agent_state.rotation, normalized_xzy)
                normalized_pt_x_in_w, normalized_pt_y_in_w, normalized_pt_z_in_w = tuple(normalized_xzy_in_w[:3])
                next_position = np.array([
                    normalized_pt_x_in_w * 0.3 + agent_state.position[0] * 0.7,
                    agent_state.position[1],
                    normalized_pt_z_in_w * 0.3 + agent_state.position[2] * 0.7
                ])

                # change rotation
                yaw_angle = np.arctan2(pt_sin, pt_cos)
                agent_rotation_vector = quaternion.as_rotation_vector(agent_state.rotation)
                agent_rotation_vector[1] += yaw_angle
                next_rotation = quaternion.from_rotation_vector(agent_rotation_vector)

                obs = simulator.sim.get_observations_at(
                    position=next_position,
                    rotation=next_rotation,
                    keep_agent_at_new_pose=True
                )

                # logging
                observations['rgb'] = obs['rgb']
                demonstra_traj.append(get_agent_map_coord(simulator.agent.state.position, info, simulator))
    
            if params['mode'] == 'manual':
                keystroke = cv2.waitKey(0)
                if chr(keystroke) in simulator.actions.keys():
                    action, log_action = simulator.actions[chr(keystroke)]
                    print(f"actions: {log_action}")
                else:
                    print("INVALID KEY")
                    continue

                observations = simulator.env.step(action)
            
            count_steps += 1
            # Update context_queue
            context_queue.pop(0)
            context_queue.append(observations['rgb'])
 
            # Update topomap node.
            closest_node += start
            reached_goal = closest_node >= (goal_node - 2)
            # if reached_goal or count_steps >= 250:

            if reached_goal or count_steps >= 150:
                print("Reached goal Stopping...")
                simulator.env.step(simulator.actions['f'][0])

    print("Episode finished after {} steps.".format(count_steps))

    if params['mode'] == 'video':
        # logging the demonstration video.
        images_to_video(
            vis_frames, params['logdir'], params['dir'], fps=6, quality=9
        )

    cv2.destroyAllWindows()

@hydra.main(version_base=None, config_path="../configs", config_name="gnm_config")
def main(cfg):
    print(f"Using {device}")
    print(OmegaConf.to_yaml(cfg))
    print('-'*50)

    model, model_params = get_model_and_model_params(cfg)
    topomap, rollout_traj, init_state = load_topomap_and_traj_and_init_state(cfg)
    run_simulation_loop(cfg, model, model_params, topomap, rollout_traj, init_state)


if __name__ == "__main__":
    main()