import math

import habitat_sim
from habitat_sim import bindings as hsim

from habitat_sim.agent.agent import AgentConfiguration

import numpy as np
import magnum as mn
import quaternion


def get_sensor_spec(agent, sensor):
    for spec in agent.agent_config.sensor_specifications:
        if spec.uuid == sensor:
            return spec
    raise ValueError()


def get_world_to_sensor_transform(agent: habitat_sim.Agent, sensor):
    """
    Camera coordinate system: x pointing to the right, y pointing up, and z pointing back
    Ref: https://aihabitat.org/docs/habitat-sim/coordinate-frame-tutorial.html#camera-coordinate-frame
    :param agent:
    :param sensor:
    :return:
    """
    sensor_state = agent.get_state().sensor_states[sensor]
    quaternion_0 = sensor_state.rotation
    translation_0 = sensor_state.position
    rotation_0 = quaternion.as_rotation_matrix(quaternion_0)
    sensor_to_world = np.eye(4)
    sensor_to_world[0:3, 0:3] = rotation_0
    sensor_to_world[0:3, 3] = translation_0
    world_to_sensor = np.linalg.inv(sensor_to_world)
    return world_to_sensor


def axisAngle2quaternion(rotation_axis, yaw_angle):
    """
    Args:
    yaw_angle: rotation angle in radian. 
    rotation_axis: (x, y, z) where x, y, z defines the axis of rotation.
    return quaternion.quaternion
    """
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    x, y, z =  tuple(rotation_axis)
    w = yaw_angle

    q_0 = np.cos(w / 2)
    q_1 = x * np.sin(w / 2)
    q_2 = y * np.sin(w / 2)
    q_3 = z * np.sin(w / 2)

    return quaternion.quaternion(np.quaternion(q_0, q_1, q_2, q_3))

def get_intrinsics(agent, sensor):
    spec = get_sensor_spec(agent, sensor)
    h, w = spec.resolution
    hfov = np.deg2rad(float(spec.hfov))

    # FIXME: assume equal focal length in x and y direction
    f = w * 0.5 / np.tan(hfov / 2.)
    K = np.array([
        [f, 0., w / 2],
        [0., f, h / 2],
        [0., 0., 1]])
    return K


def get_pixel_location(agent: habitat_sim.Agent, sensor, point):
    """
    Camera coordinate system: x pointing to the right, y pointing up, and z pointing back
    Hence anything in front of the camera has negative z coordinate. When converting to
    pixel coordinates, we follow the convention that the z coordinate is positive.
    :param agent:
    :param sensor:
    :param point:
    :return: pixel location x, y and depth d
    """
    spec = get_sensor_spec(agent, sensor)
    h, w = spec.resolution
    K = get_intrinsics(agent, sensor)

    world_to_cam = get_world_to_sensor_transform(agent, sensor)
    point_local = world_to_cam[:3, :3] @ point + world_to_cam[:3, 3]

    xyw = K @ point_local
    x, y = xyw[:2] / xyw[2]

    return w - x, y, -xyw[2]


def config_sim(scene_filepath, img_size, hfov, sensor_height=1.0, device_id=0, n_agent=1,
               dataset_config=None):
    settings = {
        "width": img_size[0],
        "height": img_size[1],
        "scene": scene_filepath,  # Scene path
        "default_agent": 0,
        "sensor_height": sensor_height,  # In meters
        "color_sensor": True,  # RGBA sensor
        "semantic_sensor": True,  # Semantic sensor
        "depth_sensor": True,  # Depth sensor
        "silent": True,
    }

    sim_cfg = hsim.SimulatorConfiguration()
    sim_cfg.enable_physics = True
    sim_cfg.gpu_device_id = device_id
    sim_cfg.scene_id = settings["scene"]
    if dataset_config is not None:
        sim_cfg.scene_dataset_config_file = dataset_config
    sim_cfg.load_semantic_mesh = True if settings["semantic_sensor"] else False

    # define default sensor parameters (see src/esp/Sensor/Sensor.h)
    sensor_specs = []
    if settings["color_sensor"]:
        color_sensor_spec = habitat_sim.CameraSensorSpec()
        color_sensor_spec.uuid = "color_sensor"
        color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_sensor_spec.hfov = hfov
        color_sensor_spec.resolution = [settings["height"], settings["width"]]
        color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(color_sensor_spec)

    if settings["depth_sensor"]:
        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.hfov = hfov
        depth_sensor_spec.resolution = [settings["height"], settings["width"]]
        depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(depth_sensor_spec)

    if settings["semantic_sensor"]:
        semantic_sensor_spec = habitat_sim.CameraSensorSpec()
        semantic_sensor_spec.uuid = "semantic_sensor"
        semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_sensor_spec.hfov = hfov
        semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
        semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(semantic_sensor_spec)

    # create agent specifications
    agent_cfgs = []
    for i in range(n_agent):
        agent_cfg = AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfgs.append(agent_cfg)

    return habitat_sim.Configuration(sim_cfg, agent_cfgs)


class SceneRenderer(object):
    def __init__(self, scene_file, img_size, hfov, sensor_height, n_agent,
                 dataset_config=None, device_id=0, uuid=''):
        """
        :param scene_file: path to the scene file
        :param img_size: a tuple (width, height)
        :param hfov: horizontal field of view in degrees
        :param n_agent: number of agents. Each agent controls a virtual camera.
        :param render_depth: if True will return the depth image
        :param uuid: a unique identifier used for creating shared memory buffers.
        """
        super(SceneRenderer, self).__init__()
        cfg = config_sim(scene_file, img_size, hfov, sensor_height, device_id, n_agent, dataset_config)
        sim = habitat_sim.Simulator(cfg)

        self.scene_file = scene_file
        self.n_agent = n_agent
        self.uuid = uuid
        self.img_size = img_size

        self.sim = sim
        self.sensor_specs = {
            k: get_sensor_spec(sim.agents[0], k)
            for k in ['color_sensor', 'depth_sensor']
        }

        self.objects = dict()

    def set_camera_pose(self, agent_id, position, orientation):
        """
        :param agent_id:
        :param position: (x, y, z)
        :param orientation: (w, x, y, z)
        :return:
        """
        agent = self.sim.agents[agent_id]
        agent.scene_node.translation = mn.Vector3(position)
        w, x, y, z = (float(_) for _ in orientation)
        agent.scene_node.rotation = mn.Quaternion((x, y, z), w)

    def get_camera_pose(self, agent_id):
        agent = self.sim.agents[agent_id]
        s = agent.get_state()
        return s.position, quaternion.as_float_array(s.rotation)

    def get_pixel_location(self, agent_id, sensor, point):
        agent = self.sim.agents[agent_id]
        return get_pixel_location(agent, sensor, point)

    def render(self, agent_id, depth=False, semantic=False):
        sim = self.sim
        obs = sim.get_sensor_observations(agent_id)
        ob = obs['color_sensor'][:, :, :3]
        ret = dict()

        ret['rgb'] = ob
        if depth:
            ret['depth'] = obs['depth_sensor']
        if semantic:
            ret['semantic'] = obs['semantic_sensor']
        return ret

    def render2(self, agent_id, pos, orn, **kwargs):
        self.set_camera_pose(agent_id, pos, orn)
        return self.render(agent_id, **kwargs)

    def find_closest_to_image_center(self, agent_id, points):
        center = np.array(self.img_size) / 2
        marker_pixel_locations = np.array(
            [self.get_pixel_location(agent_id, 'color_sensor', _) for _ in points])
        dists = np.linalg.norm(marker_pixel_locations - center[None], axis=-1)
        return marker_pixel_locations, np.argmin(dists)

    def is_navigable(self, pos):
        return self.sim.pathfinder.is_navigable(pos, 1.5)  # TODO: remove hardcoded max_y_delta

    def get_random_navigable_point(self):
        while True:
            p = self.sim.pathfinder.get_random_navigable_point()
            if not math.isnan(p[0]):
                return p
