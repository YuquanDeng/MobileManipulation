"""
Python script for deploying GNM model to Spot robot.
"""
# UTILS
import logging
import threading
from PIL import Image, ImageDraw, ImageFont
import math

# pytorch
import torch
import torch.nn as nn
from torchvision import transforms
import yaml

import numpy as np
from PIL import Image as PILImage
from typing import List

# Data Acquisition
from multiprocessing import Barrier, Process, Queue, Value
from threading import BrokenBarrierError, Thread
import time
import signal
from queue import Empty, Full
from bosdyn.api.image_pb2 import ImageSource
from bosdyn.client.frame_helpers import (GROUND_PLANE_FRAME_NAME, VISION_FRAME_NAME, BODY_FRAME_NAME, get_a_tform_b,
                                         get_vision_tform_body)
import io

# MODELS
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "../../third_party/drive-any-robot/train"))
from gnm_train.models.gnm import GNM
from gnm_train.models.stacked import StackedModel
from gnm_train.models.siamese import SiameseModel

# LOGGING & CONFIG 
from omegaconf import DictConfig, OmegaConf
import hydra

# SPOT SDK
import bosdyn.client.util
from bosdyn.client import ResponseError, RpcError, create_standard_sdk
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.estop import EstopClient
from bosdyn.client.power import PowerClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.robot_command import RobotCommandClient
from bosdyn.client.async_tasks import AsyncGRPCTask, AsyncPeriodicQuery, AsyncTasks
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import (CommandFailedError, CommandTimedOutError,
                                         RobotCommandBuilder, RobotCommandClient, blocking_stand)
from bosdyn.api import geometry_pb2 as geo
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn import geometry
from bosdyn.api import trajectory_pb2
from bosdyn.client.math_helpers import Quat, SE3Pose



MODEL_CONFIG_PATH = os.path.join(os.getcwd(), "../drive-any-robot/deployment/config/models.yaml")
MODEL_WEIGHTS_PATH = os.path.join(os.getcwd(), "../drive-any-robot/deployment/model_weights")
TOPOMAP_IMAGES_DIR = "./data"

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

# GLOBALS 
# Load the model (locobot uses a NUC, so we can't use a GPU)
device = torch.device("cpu")

LOGGER = bosdyn.client.util.get_logger()
SHUTDOWN_FLAG = Value('i', 0)
# Source List: 
#[ 'back_fisheye_image', 
# 'frontleft_fisheye_image', 
# 'frontright_fisheye_image', 
# 'hand_color_image', 
# 'hand_color_in_hand_depth_frame', 
# 'hand_image', 
# 'left_fisheye_image', 
# 'right_fisheye_image'
# ]
SOURCE_LIST = ['hand_color_image', 'hand_image']



# Context observations as input for GNM model.
context_queue = []
# The corresponding transformation for the observation.
transformation_queue = []



# Don't let the queues get too backed up
QUEUE_MAXSIZE = 5
# {
#     'source': Name of the camera,
#     'world_tform_cam': transform from VO to camera,
#     'world_tform_gpe':  transform from VO to ground plane,
#     'raw_image_time': Time when the image was collected,
#     'cv_image': The decoded image,
#     'visual_dims': (cols, rows),
#     'system_cap_time': Time when the image was received by the main process,
#     'image_queued_time': Time when the image was done preprocessing and queued
# }
RAW_IMAGES_QUEUE = Queue(QUEUE_MAXSIZE)

def signal_handler(signal, frame):
    print('Interrupt caught, shutting down')
    SHUTDOWN_FLAG.value = 1

def to_numpy(tensor):
    return tensor.cpu().detach().numpy()

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

def load_topomap(params):
    # Load topo map from a directory with RGB images collected from trajectory. 
    topomap_filenames = sorted(os.listdir(os.path.join(
    TOPOMAP_IMAGES_DIR, params['dir'])), key=lambda x: int(x.split(".")[0]))
    topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{params['dir']}"
    num_nodes = len(os.listdir(topomap_dir))
    topomap = []
    for i in range(num_nodes):
        image_path = os.path.join(topomap_dir, topomap_filenames[i])
        topomap.append(PILImage.open(image_path))
    print("-"*100)
    print("dir: ", params['dir'], "num_nodes: ", num_nodes)
    print("-"*100)

    return topomap

class AsyncRobotState(AsyncPeriodicQuery):
    """Grab robot state."""

    def __init__(self, robot_state_client):
        super(AsyncRobotState, self).__init__("robot_state", robot_state_client, LOGGER,
                                              period_sec=0.2)

    def _start_query(self):
        return self._client.get_robot_state_async()

class AsyncImageCapture(AsyncGRPCTask):
    """Grab camera images from the robot."""

    def __init__(self, robot):
        super(AsyncImageCapture, self).__init__()
        self._image_client = robot.ensure_client(ImageClient.default_service_name)
        self._ascii_image = None
        self._video_mode = False
        self._should_take_image = False

    @property
    def ascii_image(self):
        """Return the latest captured image as ascii."""
        return self._ascii_image

    def toggle_video_mode(self):
        """Toggle whether doing continuous image capture."""
        self._video_mode = not self._video_mode

    def take_image(self):
        """Request a one-shot image."""
        self._should_take_image = True

    def _start_query(self):
        self._should_take_image = False
        source_name = "frontright_fisheye_image"
        return self._image_client.get_image_from_sources_async([source_name])

    def _should_query(self, now_sec):  # pylint: disable=unused-argument
        return self._video_mode or self._should_take_image

    def _handle_result(self, result):
        import io
        image = Image.open(io.BytesIO(result[0].shot.image.data))
        self._ascii_image = _image_to_ascii(image, new_width=70)

    def _handle_error(self, exception):
        LOGGER.exception("Failure getting image: %s" % exception)

class AsyncImage(AsyncPeriodicQuery):
    """Grab image."""

    def __init__(self, image_client, image_sources):
        # Period is set to be about 15 FPS
        super(AsyncImage, self).__init__('images', image_client, LOGGER, period_sec=0.067)
        self.image_sources = image_sources

    def _start_query(self):
        return self._client.get_image_from_sources_async(self.image_sources)
    
def _update_thread(async_task):
    while True:
        async_task.update()
        time.sleep(0.01)

def capture_images(image_task, sleep_between_capture):
    while not SHUTDOWN_FLAG.value:
        get_im_resp = image_task.proto
        start_time = time.time()
        if not get_im_resp:
            continue
        raw_queue_entry = {}

        for im_resp in get_im_resp:
            if im_resp.source.image_type == ImageSource.IMAGE_TYPE_VISUAL:
                source = im_resp.source.name
                acquisition_time = im_resp.shot.acquisition_time
                image_time = acquisition_time.seconds + acquisition_time.nanos * 1e-9

                try:
                    image = Image.open(io.BytesIO(im_resp.shot.image.data))
                    source = im_resp.source.name
                    tform_snapshot = im_resp.shot.transforms_snapshot
                    frame_name = im_resp.shot.frame_name_image_sensor
                    world_tform_cam = get_a_tform_b(tform_snapshot, VISION_FRAME_NAME, frame_name)
                    world_tform_gpe = get_a_tform_b(tform_snapshot, VISION_FRAME_NAME,
                                                    GROUND_PLANE_FRAME_NAME)
                    raw_queue_entry[source] = {
                        'source': source,
                        'world_tform_cam': world_tform_cam,
                        'world_tforcm_gpe': world_tform_gpe,
                        'raw_image_time': image_time,
                        'cv_image': image,
                        'visual_dims': (im_resp.shot.image.cols, im_resp.shot.image.rows),
                        'system_cap_time': start_time,
                        'image_queued_time': time.time()
                    }

                except Exception as exc:  # pylint: disable=broad-except
                    print(f'Exception occurred during image capture {exc}')
        try:
            RAW_IMAGES_QUEUE.put_nowait(raw_queue_entry)
        except Full as exc:
            print(f'RAW_IMAGES_QUEUE is full: {exc}')
        time.sleep(sleep_between_capture)

class MobileAgent():
    """An interface of deploying GNM to Spot robot."""
    def __init__(self, params) -> None:
        """GNM and robot Spot sdk Initialization."""
        # WARNING: the correctness of the constructor depends on get_model_and_model_params() and load_topomap().

        # GNM setup.
        self._params = params
        self._model, self._model_params = get_model_and_model_params(params)
        self._topomap = load_topomap(params)

        # robot SDK setup.
        sdk = bosdyn.client.create_standard_sdk('ImageClient')
        robot = sdk.create_robot(params.hostname)
        robot.authenticate(username="admin", password="dryshyyucivp")
        # Time sync is necessary so that time-based filter requests can be converted
        robot.time_sync.wait_for_sync()

        self._robot = robot
        self._robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        self._robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        self._lease_client = robot.ensure_client(LeaseClient.default_service_name)
        self._image_client = robot.ensure_client(ImageClient.default_service_name)
        self._source_list = SOURCE_LIST  # only retrieve from the hand camera for now.
        
        # Data Acquisition setup.
        self._image_task = AsyncImage(self._image_client, self._source_list)
        self._robot_state_task = AsyncRobotState(self._robot_state_client)
        self._task_list = [self._image_task, self._robot_state_task]
        self._async_tasks = AsyncTasks(self._task_list)
        print('Image Acquisition client connected.')
    
    def start(self):
        # Verify the robot is not estopped and that an external application has registered and holds
        # an estop endpoint.
        assert not self._robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client," \
                                        " such as the estop SDK example, to configure E-Stop."

        # Start Data acquisition process.
        # This thread starts the async tasks for image and robot state retrieval.
        update_thread = Thread(target=_update_thread, args=[self._async_tasks])
        update_thread.daemon = True
        update_thread.start()
        # Wait for the first responses.
        while any(task.proto is None for task in self._task_list):
            time.sleep(0.1)

        # Start image capture process.
        image_capture_thread = Thread(target=self._capture_images,
                                       args=[],
                                       daemon=True)
        image_capture_thread.start()

    def _get_mobility_params(self):
        """Gets mobility parameters for following"""
        desired_vel = self._params['VELOCITY_BASE_ANGULAR']
        speed_limit = geo.SE2VelocityLimit(
            max_vel=geo.SE2Velocity(linear=geo.Vec2(x=desired_vel, y=desired_vel), 
                                    angular=self._params['VELOCITY_BASE_ANGULAR']))
        body_control = self._set_default_body_control()
        mobility_params = spot_command_pb2.MobilityParams(vel_limit=speed_limit, obstacle_params=None,
                                                        body_control=body_control,
                                                        locomotion_hint=spot_command_pb2.HINT_TROT)
        return mobility_params
    
    def _set_default_body_control(self):
        """Set default body control params to current body position"""
        footprint_R_body = geometry.EulerZXY()
        position = geo.Vec3(x=0.0, y=0.0, z=0.0)
        rotation = footprint_R_body.to_quaternion()
        pose = geo.SE3Pose(position=position, rotation=rotation)
        point = trajectory_pb2.SE3TrajectoryPoint(pose=pose)
        traj = trajectory_pb2.SE3Trajectory(points=[point])
        return spot_command_pb2.BodyControlParams(base_offset_rt_footprint=traj)

    def run_navigation_loop(self):
        # Lease acquiring.
        self._lease = self._lease_client.take()
        self._lease_keep = LeaseKeepAlive(self._lease_client)
        # Power on the robot and stand it up
        resp = self._robot.power_on()
        try:
            blocking_stand(self._robot_command_client)
        except CommandFailedError as exc:
            print(f'Error ({exc}) occurred while trying to stand. Check robot surroundings.')
            return False
        except CommandTimedOutError as exc:
            print(f'Stand command timed out: {exc}')
            return False
        print('Robot powered on and standing.')
        # WARNING: The mobility params depend on the Spot's current position and heading.
        mobility_params = self._get_mobility_params() 

        # Node Initialization.
        closest_node = 0
        assert -1 <= self._params['goal_node'] < len(self._topomap), "Invalid goal index"
        if self._params['goal_node'] == -1:
            goal_node = len(self._topomap) - 1
        else:
            goal_node = self._params['goal_node']
        reached_goal = False
        
        topomap = self._topomap

        # Navigation Loop.
        while not SHUTDOWN_FLAG.value:
            # continue if no enough context observations.
            if len(context_queue) < self._model_params['context'] + 1:
                time.sleep(self._params['COMMAND_INPUT_RATE'])
                continue
        
            start = max(closest_node - self._params["radius"], 0)
            end = min(closest_node + self._params["radius"] + 1, goal_node)
            distances = []
            waypoints = []
            
            for sg_img in topomap[start: end + 1]:
                transf_obs_img = transform_images(context_queue, model_params["image_size"])
                transf_sg_img = transform_images(sg_img, model_params["image_size"])
                dist, waypoint = self._model(transf_obs_img, transf_sg_img)
                distances.append(to_numpy(dist[0]))
                waypoints.append(to_numpy(waypoint[0]))

            # look for closest node
            closest_node = np.argmin(distances)
            # chose subgoal and output waypoints
            if distances[closest_node] > self._params['close_threshold']:
                chosen_waypoint = waypoints[closest_node][self._params['waypoint']]
            else:
                chosen_waypoint = waypoints[min(
                    closest_node + 1, len(waypoints) - 1)][self._params['waypoint']]
                
            if self._model_params["normalize"]:
                rate = self._params['VELOCITY_BASE_SPEED'] * self._params["sleep_between_capture"]
                chosen_waypoint[:2] *= rate

            pt_x, pt_y, pt_cos, pt_sin = tuple(chosen_waypoint)

            # waypoints conversion. (pretend the local waypoint relative to body frame)
            body_tform_waypoint = SE3Pose(pt_x, pt_y, 0, Quat().from_yaw(np.arctan2(pt_sin, pt_cos)))
            robot_state = self._robot_state_task.proto
            world_tform_body = get_vision_tform_body(robot_state.kinematic_state.transforms_snapshot)
            world_tform_waypoint = world_tform_body * body_tform_waypoint

            # Build and send command.
            cmd = RobotCommandBuilder.trajectory_command(goal_x=world_tform_waypoint.x,
                                                         goal_y=world_tform_waypoint.y,
                                                         goal_heading=world_tform_waypoint.rot.to_yaw(),
                                                         frame_name=VISION_FRAME_NAME,
                                                         params=mobility_params
                                                         )
            
            end_time = 15.0
            self._robot_command_client.robot_command(lease=None, command=cmd,
                                                     end_time_secs=time.time() + end_time)
            
            # Update topomap node.
            closest_node += start
            reached_goal = closest_node == goal_node

            # For debugging.
            for i in range(len(distances)):
                print(f" {i}: {distances[i]}", end='|')  
            print()
            print()
            concatenated_imgs = self.concatenate_images_horizontally(context_queue[-1], 
                                                                     topomap[closest_node],
                                                                     closest_node,
                                                                     chosen_waypoint, 
                                                                     start)
            concatenated_imgs.save('./data/real_obs.jpg')

            # Check reach goal.
            if reached_goal:
                print("closest node", closest_node)
                print("Reached goal Stopping...")
                return
            
            time.sleep(1)
        
        print("Shutting down the agent...")
        self._lease_keep.shutdown()
        self._lease_client.return_lease(self._lease)


    def get_go_to(self, world_tform_object, robot_state, mobility_params, dist_margin=0.5):
        """Gets trajectory command to a goal location

        Args:
            world_tform_object (SE3Pose): Transform from vision frame to target object
            robot_state (RobotState): Current robot state
            mobility_params (MobilityParams): Mobility parameters
            dist_margin (float): Distance margin to target
        """
        vo_tform_robot = get_vision_tform_body(robot_state.kinematic_state.transforms_snapshot)
        print(f'robot pos: {vo_tform_robot}')
        delta_ewrt_vo = np.array(
            [world_tform_object.x - vo_tform_robot.x, world_tform_object.y - vo_tform_robot.y, 0])
        norm = np.linalg.norm(delta_ewrt_vo)
        if norm == 0:
            return None
        delta_ewrt_vo_norm = delta_ewrt_vo / norm
        heading = self._get_heading(delta_ewrt_vo_norm)
        vo_tform_goal = np.array([
            world_tform_object.x - delta_ewrt_vo_norm[0] * dist_margin,
            world_tform_object.y - delta_ewrt_vo_norm[1] * dist_margin
        ])
        tag_cmd = RobotCommandBuilder.trajectory_command(goal_x=vo_tform_goal[0],
                                                        goal_y=vo_tform_goal[1], 
                                                        goal_heading=vo_tform_robot.rot.to_yaw(),
                                                        frame_name=VISION_FRAME_NAME,
                                                        params=mobility_params)
        return tag_cmd

    def _test_world_tform_cam(self):
         # Lease acquiring.
        self._lease = self._lease_client.take()
        self._lease_keep = LeaseKeepAlive(self._lease_client)
        # Power on the robot and stand it up
        resp = self._robot.power_on()

        try:
            blocking_stand(self._robot_command_client)
        except CommandFailedError as exc:
            print(f'Error ({exc}) occurred while trying to stand. Check robot surroundings.')
            return False
        except CommandTimedOutError as exc:
            print(f'Stand command timed out: {exc}')
            return False
        print('Robot powered on and standing.')
        # WARNING: The mobility params depend on the Spot's current position and heading.
        mobility_params = self._get_mobility_params() 

        # world -> body -> waypoint.
        while not SHUTDOWN_FLAG.value:
            # waypoints conversion.
            body_tform_waypoint = SE3Pose(-0.2, 0, 0, Quat()) 
            robot_state = self._robot_state_task.proto
            world_tform_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot, 
                                             VISION_FRAME_NAME,
                                             BODY_FRAME_NAME)
            world_tform_waypoint = world_tform_body * body_tform_waypoint
            cmd = RobotCommandBuilder.trajectory_command(goal_x=world_tform_waypoint.x,
                                                         goal_y=world_tform_waypoint.y,
                                                         goal_heading=world_tform_waypoint.rot.to_yaw(),
                                                         frame_name=VISION_FRAME_NAME,
                                                         params=mobility_params)
            end_time = 15.0
            self._robot_command_client.robot_command(lease=None, command=cmd,
                                                     end_time_secs=time.time() + end_time)

            time.sleep(3)

        # Shutdown lease keep-alive and return lease gracefully.
        print("Shutting down the agent...")
        self._lease_keep.shutdown()
        self._lease_client.return_lease(self._lease)
        return 
    
    def _get_heading(self, xhat):
        zhat = [0.0, 0.0, 1.0]
        yhat = np.cross(zhat, xhat)
        mat = np.array([xhat, yhat, zhat]).transpose()
        return Quat.from_matrix(mat).to_yaw()


    # TODO: Delete this after debugging. 
    def concatenate_images_horizontally(self, img1, img2, closest_node, chosen_waypoint, start):
        width = img1.width + img2.width
        height = max(img1.height, img2.height)
        yaw_angle = math.degrees(np.arctan2(chosen_waypoint[-1], chosen_waypoint[-2]))

        new_img = Image.new('RGB', (width, height))
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (img1.width, 0))


        # Draw observation image text.
        draw = ImageDraw.Draw(new_img)
        font_size = 70  # You can adjust the font size as needed
        font = ImageFont.truetype("arial.ttf", font_size)  # Change the font and path as needed
        text_color = (255, 255, 255)  # White color, you can change it to any other RGB value
        text_position = (width/7, height * (3/4))  # Adjust the (x, y) position where you want to put the text
        draw.text(text_position, "curr obs", font=font, fill=text_color)
        text_position = (width * (0.1/7), height * (0.1/4))
        draw.text(text_position, f"waypoint: {chosen_waypoint[:2]}",  
                  font=ImageFont.truetype("arial.ttf", 20), fill=text_color)
        
        # Draw yaw angle text.
        text_position = (width * (0.1/7), height * (0.3/4))
        draw.text(text_position, f"yaw_angle(in degree): {yaw_angle}",  
                  font=ImageFont.truetype("arial.ttf", 20), fill=text_color)
        
        # Draw closest node text.
        text_position = (width * (0.1/7), height * (0.5/4))
        draw.text(text_position, f"closest node: {closest_node}",  
                  font=ImageFont.truetype("arial.ttf", 20), fill=text_color)
        
        # Draw start node text.
        text_position = (width * (0.1/7), height * (0.7/4))
        draw.text(text_position, f"start node: {start}",  
                  font=ImageFont.truetype("arial.ttf", 20), fill=text_color)

        # local cloest node
        text_position = (width * (0.1/7), height * (0.9/4))
        draw.text(text_position, f"local closest node: {closest_node-start}",  
        font=ImageFont.truetype("arial.ttf", 20), fill=text_color)


        # Draw short goal image text.
        text_position = (width * (4/7), height * (3/4))
        draw.text(text_position, f"short goal: {closest_node}", font=font, fill=text_color)
        return new_img
    
    def _capture_images(self):
        "Grab images from robot Spot."
        while not SHUTDOWN_FLAG.value:
            get_im_resp = self._image_task.proto
            if not get_im_resp:
                continue
            
            # Invariant: each protocal buffer should have and only have 1 image satisfied.
            for im_resp in get_im_resp:
                if im_resp.source.name == self._params['OBSERVATION_SOURCE']:
                    try:
                        tform_snapshot = im_resp.shot.transforms_snapshot
                        frame_name = im_resp.shot.frame_name_image_sensor
                        world_tform_cam = get_a_tform_b(tform_snapshot, VISION_FRAME_NAME, frame_name)
                    except Exception as exc:  # pylint: disable=broad-except
                        print(f'Exception occurred during image capture {exc}')
                if im_resp.source.name == 'hand_color_image':
                    obs_img = Image.open(io.BytesIO(im_resp.shot.image.data))
            # For past observation.
            if len(context_queue) < self._model_params["context"] + 1:
                context_queue.append(obs_img)
            else:
                context_queue.pop(0)
                context_queue.append(obs_img)

            # For transformation matrix
            if len(transformation_queue) < self._model_params["context"] + 1:
                transformation_queue.append(world_tform_cam)
            else:
                transformation_queue.pop(0)
                transformation_queue.append(world_tform_cam)
                
            time.sleep(self._params.sleep_between_capture)

    def shutdown(self):
        
        return True

@hydra.main(version_base=None, config_path="../conf", config_name="spot_config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    print("-"*50)

    signal.signal(signal.SIGINT, signal_handler)
    try:
        mobile_agent = MobileAgent(cfg)
        mobile_agent.start()
        mobile_agent.run_navigation_loop()
        # mobile_agent._test_world_tform_cam()
    except Exception as exc:
        LOGGER.error("Spot threw an exception: %s", exc)
        return False
    return True

if __name__ == "__main__":
    if not main():
        os._exit(1)
    os._exit(0)