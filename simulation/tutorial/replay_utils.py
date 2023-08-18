import os
import sys

import git
import magnum as mn
import numpy as np

import habitat_sim
from habitat_sim.gfx import LightInfo, LightPositionModel
from habitat_sim.utils import gfx_replay_utils
from habitat_sim.utils import viz_utils as vut

dir_path = os.getcwd()
data_path = os.path.join(dir_path, "../data")
output_path = os.path.join(dir_path, "replay_tutorial_output/")

def make_configuration(settings):
    make_video_during_sim = False
    if "make_video_during_sim" in settings:
        make_video_during_sim = settings["make_video_during_sim"]

    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = os.path.join(
        data_path, "scene_datasets/habitat-test-scenes/apartment_1.glb"
    )
    assert os.path.exists(backend_cfg.scene_id)
    backend_cfg.enable_physics = True

    # Enable gfx replay save. See also our call to sim.gfx_replay_manager.save_keyframe()
    # below.
    backend_cfg.enable_gfx_replay_save = True
    backend_cfg.create_renderer = make_video_during_sim

    sensor_cfg = habitat_sim.CameraSensorSpec()
    sensor_cfg.resolution = [544, 720]
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_cfg]

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])

def simulate_with_moving_agent(
    sim,
    duration=1.0,
    agent_vel=np.array([0, 0, 0]),
    look_rotation_vel=0.0,
    get_frames=True,
):
    sensor_node = sim._sensors["rgba_camera"]._sensor_object.object
    agent_node = sim.get_agent(0).body.object

    # simulate dt seconds at 60Hz to the nearest fixed timestep
    time_step = 1.0 / 60.0

    rotation_x = mn.Quaternion.rotation(
        mn.Deg(look_rotation_vel) * time_step, mn.Vector3(1.0, 0, 0)
    )

    print("Simulating " + str(duration) + " world seconds.")
    observations = []
    start_time = sim.get_world_time()
    while sim.get_world_time() < start_time + duration:
        # move agent
        agent_node.translation += agent_vel * time_step

        # rotate sensor
        sensor_node.rotation *= rotation_x

        # Add user transforms for the agent and sensor. We'll use these later during
        # replay playback.
        gfx_replay_utils.add_node_user_transform(sim, agent_node, "agent")
        gfx_replay_utils.add_node_user_transform(sim, sensor_node, "sensor")

        sim.step_physics(time_step)

        # save a replay keyframe after every physics step
        sim.gfx_replay_manager.save_keyframe()

        if get_frames:
            observations.append(sim.get_sensor_observations())

    return observations

def configure_lighting(sim):
    light_setup = [
        LightInfo(
            vector=[1.0, 1.0, 0.0, 1.0],
            color=[18.0, 18.0, 18.0],
            model=LightPositionModel.Global,
        ),
        LightInfo(
            vector=[0.0, -1.0, 0.0, 1.0],
            color=[5.0, 5.0, 5.0],
            model=LightPositionModel.Global,
        ),
        LightInfo(
            vector=[-1.0, 1.0, 1.0, 1.0],
            color=[18.0, 18.0, 18.0],
            model=LightPositionModel.Global,
        ),
    ]
    sim.set_light_setup(light_setup)

