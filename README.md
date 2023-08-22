# Mobile Manipulation

## Overview
This repository contains code of deploying GNM(refer to General Navigation Model in the paper [GNM: A General Navigation Model to Drive Any Robot](https://arxiv.org/abs/2210.03370)) in [habitat](https://aihabitat.org/) simulation environment and the robot [Spot](https://bostondynamics.com/products/spot/). The repository has the following structure:

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


## Installation
Since habitat simulation requires python version at least higher than 3.9 while spot-sdk only supports python version 3.6-3.8, unfortunately we have to create two virtual environments for the setup.
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


