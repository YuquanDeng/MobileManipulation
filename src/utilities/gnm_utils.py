# Habitat sim
# HABITAT SIM
import habitat
import cv2
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations import maps
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_rotate_vector,
)


# pytorch
import torch
import torch.nn as nn
from torchvision import transforms

import numpy as np
from PIL import Image as PILImage
from typing import List

# models
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "../drive-any-robot/train"))

from gnm_train.models.gnm import GNM
from gnm_train.models.stacked import StackedModel
from gnm_train.models.siamese import SiameseModel

# top down map configs
MAP_THICKNESS_SCALAR: int = 128

"""
Contains methods that used for deploying GNM model into Habitat simulation environment.
"""


# Key response for agent actions.
keys = {
    "FORWARD_KEY": "w", 
    "LEFT_KEY": "a",
    "RIGHT_KEY": "d", 
    "SAVE_KEY": "s",
    "FINISH": "f"
}

def load_model(
    model_path: str,
    model_type: str,
    context: int,
    len_traj_pred: int,
    learn_angle: bool,
    obs_encoding_size: int = 1024,
    goal_encoding_size: int = 1024,
    obsgoal_encoding_size: int = 2048,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Load a model from a checkpoint file (works with models trained on multiple GPUs)"""
    checkpoint = torch.load(model_path, map_location=device)
    loaded_model = checkpoint["model"]
    if model_type == "gnm":
        model = GNM(
            context,
            len_traj_pred,
            learn_angle,
            obs_encoding_size,
            goal_encoding_size,
        )
    elif model_type == "siamese":
        model = SiameseModel(
            context,
            len_traj_pred,
            learn_angle,
            obs_encoding_size,
            goal_encoding_size,
            obsgoal_encoding_size,
        )
    elif model_type == "stacked":
        model = StackedModel(
            context,
            len_traj_pred,
            learn_angle,
            obs_encoding_size,
            obsgoal_encoding_size,
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    try:
        state_dict = loaded_model.module.state_dict()
        model.load_state_dict(state_dict)
    except AttributeError as e:
        state_dict = loaded_model.state_dict()
        model.load_state_dict(state_dict)
    model.to(device)
    return model

def transform_images(
    pil_imgs: List[PILImage.Image], image_size: List[int]
) -> torch.Tensor:
    """
    Transforms a list of PIL image to a torch tensor.
    Args:
        pil_imgs (List[PILImage.Image]): List of PIL images to transform and concatenate
        image_size (int, int): Size of the output image [width, height]
    """
    assert len(image_size) == 2
    image_size = image_size[::-1] # torchvision's transforms.Resize expects [height, width]
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(image_size[::-1]),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        transf_img = transform_type(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)

def to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def start_sim(env):
    print("Environment creation successful")
    observations = env.reset()
    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
        observations["pointgoal_with_gps_compass"][0],
        observations["pointgoal_with_gps_compass"][1]))
    cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

    print("Agent stepping around inside environment.")
    return observations['rgb']

def get_agent_action(keystroke):
    if keystroke == ord(keys['FORWARD_KEY']):
        action = HabitatSimActions.move_forward
        print("action: FORWARD")
    elif keystroke == ord(keys['LEFT_KEY']):
        action = HabitatSimActions.turn_left
        print("action: LEFT")
    elif keystroke == ord(keys['RIGHT_KEY']):
        action = HabitatSimActions.turn_right
        print("action: RIGHT")
    elif keystroke == ord(keys['FINISH']):
        action = HabitatSimActions.stop
        print("action: FINISH")
    else:
        print("INVALID KEY")
        return None
    return action

def terminate(action, observations):
    if (
        action == HabitatSimActions.stop
        and observations["pointgoal_with_gps_compass"][0] < 0.2
    ):
        print("you successfully navigated to destination point")
    else:
        print("your navigation was unsuccessful")


# TOP DOWN MAP VISUALIZATION UTILS.
def get_waypoint(
        args, 
        closest_node, 
        goal_node, 
        context_queue,
        model,
        model_params,
        topomap):
    """
    Feed context images and short goal image into GNM and return the waypoint.
    """
    start = max(closest_node - args.radius, 0)
    end = min(closest_node + args.radius + 1, goal_node)
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
    if distances[closest_node] > args.close_threshold:
        chosen_waypoint = waypoints[closest_node][args.waypoint]
    else:
        chosen_waypoint = waypoints[min(
            closest_node + 1, len(waypoints) - 1)][args.waypoint]
    
    # TODO: Not sure if I need to do this.
    if model_params["normalize"]:
        # chosen_waypoint[:2] *= MAX_V
        pass
    return chosen_waypoint, start, end, closest_node

def set_agent_map_coord_and_angle(agent_state, info, simulator):
    # set position
    a_x, a_y = get_agent_map_coord(agent_state.position, info, simulator)
    info['top_down_map']['agent_map_coord'] = np.array([(a_x, a_y)])

    # set rotation.
    phi = get_agent_map_angle(agent_state.rotation)
    info['top_down_map']['agent_angle'] = [np.array(phi)]
    return info

def get_agent_map_coord(position, info, simulator):
    # set position
    a_x, a_y = maps.to_grid(
    position[2],
    position[0],
    (info['top_down_map']['map'].shape[0], info['top_down_map']['map'].shape[1]),
    sim=simulator.sim,
    )
    return a_x, a_y

def get_agent_map_angle(rotation):
    ref_rotation = rotation
    heading_vector = quaternion_rotate_vector(
        ref_rotation.inverse(), np.array([0, 0, -1])
    )
    phi = cartesian_to_polar(heading_vector[2], -heading_vector[0])[1]
    return phi

def draw_point(position, point_type, measurements, simulator):
    map_resolution = 1024
    point_padding = 2 * int(
        np.ceil(map_resolution / MAP_THICKNESS_SCALAR)
    )
    t_x, t_y = maps.to_grid(
        position[2],
        position[0],
        (measurements['top_down_map']['map'].shape[0], measurements['top_down_map']['map'].shape[1]),
        sim=simulator.sim,
    )
    measurements['top_down_map']['map'][
        t_x - point_padding : t_x + point_padding + 1,
        t_y - point_padding : t_y + point_padding + 1,
    ] = point_type

def draw_rollout_traj(rollout_traj, info, simulator):
    # Draw source point and target point.
    draw_point(
        position=rollout_traj[0],
        point_type= maps.MAP_SOURCE_POINT_INDICATOR,
        measurements=info,
        simulator=simulator
    )
    draw_point(
        position=rollout_traj[-1],
        point_type= maps.MAP_TARGET_POINT_INDICATOR,
        measurements=info,
        simulator=simulator
    )

    for i in range(len(rollout_traj)):
        a_x, a_y = get_agent_map_coord(position=rollout_traj[i], info=info, simulator=simulator)
        rollout_traj[i] = (a_x, a_y)

    # Draw rollout trajectory. 
    maps.draw_path(
        top_down_map=info['top_down_map']['map'],
        path_points=rollout_traj,
        color=maps.MAP_SHORTEST_PATH_COLOR,
        thickness=10,
    )

