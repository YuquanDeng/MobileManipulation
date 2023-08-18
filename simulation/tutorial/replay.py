import os
import sys

import git
import magnum as mn
import numpy as np

import habitat_sim
from habitat_sim.gfx import LightInfo, LightPositionModel
from habitat_sim.utils import gfx_replay_utils
from habitat_sim.utils import viz_utils as vut

from replay_utils import make_configuration, configure_lighting, simulate_with_moving_agent

dir_path = os.getcwd()
output_path = os.path.join(dir_path, "replay_tutorial_output/")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show-video", dest="show_video", action="store_false")
    parser.add_argument("--no-make-video", dest="make_video", action="store_false")
    parser.set_defaults(show_video=False, make_video=True)
    args, _ = parser.parse_known_args()
    show_video = args.show_video
    make_video = args.make_video
    make_video_during_sim = True

    if make_video and not os.path.exists(output_path):
        os.mkdir(output_path)

    cfg = make_configuration({"make_video_during_sim": make_video_during_sim})

    sim = None
    replay_filepath = "./replay.json"

    if not sim:
        sim = habitat_sim.Simulator(cfg)
    else:
        sim.reconfigure(cfg)

    configure_lighting(sim)

    agent_state = habitat_sim.AgentState()
    agent = sim.initialize_agent(0, agent_state)

    # Initial placement for agent and sensor
    agent_node = sim.get_agent(0).body.object
    sensor_node = sim._sensors["rgba_camera"]._sensor_object.object

    # initial agent transform
    agent_node.translation = [-0.15, -1.5, 1.0]
    agent_node.rotation = mn.Quaternion.rotation(mn.Deg(-75), mn.Vector3(0.0, 1.0, 0))

    # initial sensor local transform (relative to agent)
    sensor_node.translation = [0.0, 0.6, 0.0]
    sensor_node.rotation = mn.Quaternion.rotation(mn.Deg(-15), mn.Vector3(1.0, 0.0, 0))

    observations = []

    # simulate with empty scene
    observations += simulate_with_moving_agent(
        sim,
        duration=1.0,
        agent_vel=np.array([0.5, 0.0, 0.0]),
        look_rotation_vel=25.0,
        get_frames=make_video_during_sim,
    )

    if make_video_during_sim:
        vut.make_video(
            observations,
            "rgba_camera",
            "color",
            output_path + "episode",
            open_vid=show_video,
        )

    sim.gfx_replay_manager.write_saved_keyframes_to_file(replay_filepath)
    assert os.path.exists(replay_filepath)

    # Play Back
    sim.close()

    # use same agents/sensors from earlier, with different backend config
    playback_cfg = habitat_sim.Configuration(
        gfx_replay_utils.make_backend_configuration_for_playback(
            need_separate_semantic_scene_graph=False
        ),
        cfg.agents,
    )

    if not sim:
        sim = habitat_sim.Simulator(playback_cfg)
    else:
        sim.reconfigure(playback_cfg)

    configure_lighting(sim)

    agent_state = habitat_sim.AgentState()
    sim.initialize_agent(0, agent_state)

    # Initialize a dummy agent.
    agent_node = sim.get_agent(0).body.object
    sensor_node = sim._sensors["rgba_camera"]._sensor_object.object
    agent_node.translation = [0.0, 0.0, 0.0]
    agent_node.rotation = mn.Quaternion()

    # Load early saved replay.
    player = sim.gfx_replay_manager.read_keyframes_from_file(replay_filepath)
    assert player



    # Play the replay. 
    observations = []
    print("play replay #0...")
    for frame in range(player.get_num_keyframes()):
        player.set_keyframe_index(frame)

        (sensor_node.translation, sensor_node.rotation) = player.get_user_transform(
            "sensor"
        )

        observations.append(sim.get_sensor_observations())

    if make_video:
        vut.make_video(
            observations,
            "rgba_camera",
            "color",
            output_path + "replay_playback1",
            open_vid=show_video,
        )

    observations = []
    print("play in reverse at 3x...")
    for frame in range(player.get_num_keyframes() - 2, -1, -3):
        player.set_keyframe_index(frame)
        (sensor_node.translation, sensor_node.rotation) = player.get_user_transform(
            "sensor"
        )
        observations.append(sim.get_sensor_observations())

    if make_video:
        vut.make_video(
            observations,
            "rgba_camera",
            "color",
            output_path + "replay_playback2",
            open_vid=show_video,
        )
    