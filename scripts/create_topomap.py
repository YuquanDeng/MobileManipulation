import sys
sys.path.append("../")
import os
import shutil
from simulation.simulator import *
from infrastructure.utils import load, save
from omegaconf import DictConfig, OmegaConf


# python3 create_topomap.py  --no-auto-logging

@hydra.main(version_base=None, config_path="../conf", config_name="topomap_config")
def test(cfg):
    print(OmegaConf.to_yaml(cfg))
    print('-'*50)
    params = cfg

    # Run episode
    simulator = Simulator()
    simulator.init_env(params=cfg)
    init_position, init_rotation = get_init_state(params['init_pos'], params['init_rot'])
    if not params['no_reset_pos']:
        actions = simulator.run_sim_with_topdown_map(
            log_action=params['playback'], 
            init_pos=init_position,
            init_rotation=init_rotation
        )
    else:
        actions = simulator.run_sim_with_topdown_map(log_action=params['playback'])

    # Initialize Simulator object and call playback() method.
    if params['playback']:
        print("-"*50)
        print("in playback")
        print("-"*50)

        actions_dir = os.path.join('./data/topomap/actions/', params['dir'])
        images_dir = os.path.join('./data/topomap/images/', params['dir'])
        
        if os.path.exists(actions_dir):
            shutil.rmtree(actions_dir)

        if not os.path.exists(actions_dir):
            os.makedirs(actions_dir)

        
        if os.path.exists(images_dir):
            shutil.rmtree(images_dir)
            
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
    
        save(file=actions, filepath=os.path.join(actions_dir, params['dir']+'.pkl'))
        simulator.playback(
            actions=actions,
            topomap_name=params['dir'],
            images_dir=images_dir,
            actions_dir=actions_dir,
            init_pos=init_position,
            init_rotation=init_rotation,
            params=cfg
        )

def get_init_state(position, rotation):
    if position == 'bl':
        init_position = BOTTOM_LEFT_POS
    elif position == 'br':
        init_position = BOTTOM_RIGHT_POS
    elif position == 'ul':
        init_position = UPPER_LEFT_POS
    elif position == 'ur':
        init_position = UPPER_RIGHT_POS

    if rotation == 'up':
        init_rotation = UP_ROTATION
    elif rotation == 'left':
        init_rotation = LEFT_ROTATION
    elif rotation == 'right':
        init_rotation = RIGHT_ROTATION
    elif rotation == 'down':
        init_rotation = DOWN_ROTATION

    return init_position, init_rotation

@hydra.main(version_base=None, config_path="../conf", config_name="topomap_config")
def playback(cfg):
    actions = load(filepath=os.path.join('./data/topomap/actions/testing.pkl'))
    topomap_name = 'testing'
    simulator = Simulator()
    simulator.init_env(params=cfg)
    simulator.playback(
        actions=actions,
        topomap_name=topomap_name,
        topomap_name_dir=os.path.join('./data/topomap/images/', topomap_name),
        reset_pos=True,
        params=cfg
    )


if __name__ == "__main__":
    test()