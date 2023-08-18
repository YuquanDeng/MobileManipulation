# Mobile Manipulation

## Installation

### habitat-sim
Setup the habitat simulation environment by following the issue https://github.com/facebookresearch/habitat-sim/issues/2049

After setting up the habitat simulation environment, install the require packages as below:
<pre> <code>
cd mobile-manipulation/
conda activate habitat
pip install -r requirements.txt
</code> </pre>

### Update simulation environment
You can update the submodule to the latest version by running the command:
<pre> <code>
git submodule update --remote
</code> </pre>

Run the following command after cloned the repository.
<pre> <code>
git submodule update --init --recursive
</code> </pre>


## Testing
### Non-interactive testing

<pre> <code>
cd mobile-manipulation/simulation/
python ./habitat-sim/examples/example.py --scene ./data/scene_datasets/habitat-test-scenes/skokloster-castle.glb
</code> </pre>



## Load dataset, Train model, Save model, and Log result.
- `run_experiment.py` is the python script that contains the pipeline of loading dataset, training model, saving model, and logging results.
- `run_experiment.py` will load the configuration in `conf/config.yaml`. So far all the configs have default values, so keep in mind what the default values are when running
<pre> <code>
cd mobile-manipulation/scripts/
python3 run_experiment.py 
</code> </pre>
without specifying any change of the configurations.

- For any customized change of the configuration, we can run:
<pre> <code>
python3 run_experiment.py preprocessed_features=true preprocessed_dataset=true
</code> </pre>
