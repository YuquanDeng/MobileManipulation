# Mobile Manipulation

## Overview
This repository contains code of deploying GNM(refer to General Navigation Model in the paper [GNM: A General Navigation Model to Drive Any Robot](https://arxiv.org/abs/2210.03370)) in [habitat](https://aihabitat.org/) simulation environment and the robot [Spot](https://bostondynamics.com/products/spot/).


## Installation
Since habitat simulation requires python version at least higher than 3.9 while spot-sdk only supports python version 3.6-3.8, unfortunately we have to create two virtual environments for the setup.

- Download GNM checkpoint `gnm_large` under the directory `third_party/drive-any-robot/deployment/model_weights` from [this link](https://drive.google.com/drive/folders/1np7D0Ak7x10IoQn9h0qxn8eoxJQiw8Dr?usp=share_link).

### Simulation environment setup
1. create virtual environment `mobile_manipulation`.
<pre><code>
conda create -n mobile_manipulation python=3.9 cmake=3.14.0
conda activate mobile_manipulation
</code></pre>

2. Install `habitat-sim` and `habitat-lab`.
<pre><code>
conda install habitat-sim withbullet -c conda-forge -c aihabitat
</code></pre>

At the root of the repository,
<pre><code>
git submodule update --init --recursive
cd third_party/habitat-lab
pip install -e habitat-lab  # install habitat_lab
pip install -e habitat-baselines  # install habitat_baselines
</code></pre>

### Testing
- At the root of the repository, download testing data under the directory `data/habitat/`.
<pre><code>
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/habitat/

# Download example objects
python -m habitat_sim.utils.datasets_download --uids habitat_example_objects --data-path data/habitat/

# Download point-goal navigation episodes for the test scenes:
python -m habitat_sim.utils.datasets_download --uids habitat_test_pointnav_dataset --data-path data/habitat/
</code></pre>

- Habitat-sim testing
<pre><code>
cd third_party/habitat-sim # At the root of the repository
python examples/viewer.py --scene ../../data/habitat/scene_datasets/habitat-test-scenes/skokloster-castle.glb
</code></pre>

#NOTE: depending on your choice of installation, you may need to add '/path/to/habitat-sim' to your PYTHONPATH.
#e.g. from 'habitat-sim/' directory run 'export PYTHONPATH=$(pwd)'

- Habitat-lab testing (non-interactive)

<pre><code>
# Back to the root of the repository.

cd third_party/habitat-lab
python examples/example.py
</code></pre>

For more installation and testing details: [habitat-sim installation](https://github.com/facebookresearch/habitat-sim) and [habitat-lab](https://github.com/facebookresearch/habitat-lab) installation.

- GNM Setup
<pre><code>
pip install torch==1.11.0 torchvision==0.12.0 tqdm==4.64.0 numpy==1.24.0
conda install -c pytorch matplotlib
pip install -e third_party/drive-any-robot/train/
</code></pre>

### Spot Deployment Environment Setup
- Create another virtual environment `deployment`
<pre><code>
python3.8 -m venv deployment
source deployment/bin/activate

python3 -m pip install bosdyn-client==3.2.3 bosdyn-mission==3.2.3 bosdyn-choreography-client==3.2.3
python3 -m pip list --format=columns | grep bosdyn
</code></pre>

- GNM Setup
<pre><code>
pip install torch==1.11.0 torchvision==0.12.0 tqdm==4.64.0 numpy==1.24.0
conda install -c pytorch matplotlib
pip install -e third_party/drive-any-robot/train/
</code></pre>

## Deployment in Simulation

The recorded trajectory data will be stored under the directory `data` and follow the structure:

The repository has the following structure:

<pre> <code>
data
  ├── deployment
  |   └── &lt;name_of_traj1&gt;
  |       ├── 0.jpg
  |       ├── 1.jpg
  |       └── ...
  ├── habitat
  |   └── topomap
  |       └── &lt;name_of_traj1&gt;
  |           ├── action
  |           |   └── ...pkl
  |           └── images
  |               ├── 0.jpg
  |               ├── 1.jpg
  |               └── ... 
  └── siamese
</code> </pre>

### Create topological map in habitat simulation
- Activate `mobile_manipulation` virtual environment.
<pre><code>
conda activate mobile_manipulation
cd src/simulation
python3 create_topomap.py
</code></pre>
The `create_topomap.py` script will load the [topomap_config.yaml](https://github.com/YuquanDeng/MobileManipulation/blob/main/src/configs/topomap_config.yaml) under the directory `src/configs`. In this file, you can specify the setting of the topomap:
- `dir`(str,  default: 'testing'): The name of the logging directory of recorded topological map. 
- `playback`(bool, default: true): 
- `no_reset_pos`(bool, default: false): Resetting the initial position or not.
- `init_pos`(str, default: bl): The map is a rectangle. The initial position has four options: bottom left(bl), bottom right(br), upper left(ul), upper right(ur).
- `init_rot`(str, default: right): The rotation angle has four options: up, down, left, right.
- `resolution`(int, default: 256): resolution of the camera
- `width`(int, default: 256): The width of the image
- `height`(int, default: 256): The height of the image
- `sensor_position`(tuple, default: 0, 1.25, 0): The position of the sensor(e.g. RGB sensor or depth sensor). For the coordinate, please refer to [Coordinate Frame](https://aihabitat.org/docs/habitat-sim/coordinate-frame-tutorial.html). For the deafult case, `1.25` means the height of the sensor is 1.25 meter.
- `sensor_orientation`(tuple, default: 0, 0, 0): The orientatin of the sensor(e.g. RGB sensor or depth sensor).\
- `hfov`(int, default: 90): The field of view of the camera. The default value is 90 degrees.

`scenes_dir`: The directory of loading scenes dataset.
`data_path`: The directory of loading other habitat lab configs.
`logdir`: The directory of logging overrided configs of topomap.yaml.

The `create_topomap.py` script will also load `gnm_config.yaml`. For the details of the parameters please refer to `Running the model` in the [drive-any-robot](https://github.com/YuquanDeng/drive-any-robot) repository.

### Running the GNM in Habitat Simulation 
<pre><code>
cd src/simulation
python3 navigate.py
</code></pre>

## Deployment on the robot Spot
- Activate the virtual environment `deployment`
<pre><code>
source deployment/bin/activate
</code></pre>

### Record the trajectory
The script `capture_image.py` is a `wasd` interface to operate spot and starting capturing image by press "I" and ending capturing image by pressing "O".
<pre><code>
cd src/deployment
python3 capture_image.py
</code></pre>

- The script `capture_image.py` loads the `spot_config.yaml` under the directory `src/configs`. In this file, you can specify the parameter of the Spot:
- `hostname`(default: 192.168.80.3): The hostname of connecting to the Spot.
- `time_sync_interval_sec`(default: _): The interval between time synchronization.
- `VELOCITY_BASE_SPEED`(float, default: 0.5 m/s): The linear velocity of the spot. The default value is 0.5 m/s.
- `VELOCITY_BASE_ANGULAR`(float, default: 0.53 rad/sec): The angular velocity of the spot. The default value is 0.53 rad/sec.
- `VELOCITY_CMD_DURATION`(int, default: 1 sec): The duration of commanding the velocity. 
- `COMMAND_INPUT_RATE`(float, default: 0.1 sec): The interval of sending each command.
- `OBSERVATION_SOURCE`(str, default: 'hand_image'): The camera source of the observation images.
- 'dir'(str, default: `test`): The name of the logging directory of recorded topological map. 
- `sleep_between_capture`(float, defaut: 0.2): The interval between acquiring images.


### Running the GNM on the robot Spot
<pre><code>
cd src/deployment
python3 navigate.py
</code></pre>

## Unsolved Issues
- After testing GNM on some trajecties involved turning, the GNM doesn't work well(i.e. it takes long time to finish a short distance trajectory or get stuck with some intermediate short goal image). Fine tune the parameters doesn't help that much on the performance. The quesiton is why GNM doesn't work. For the following work, it will be good to build some visulization tools to help the understandings:
- simulate the spot configuration in Habitat and do the top-down visualization.
- visualize the top down visualization in the real world by using  http://wiki.ros.org/rtabmap_ros, an example will be in https://homes.cs.washington.edu/~xiangyun/topological_nav/sample_videos.html .
- Using the opti track to build the map.


