# Experiments for PAL based on LE codebase


## Installation

Clone all contents of this folder, cd into it.

Make new python environment and activate:
```
python3 -m venv GLEnv
source GLEnv/bin/activate
```

Install requirements:
`pip install -r requirements.txt`

Set variable by adding to `~/.bashrc`:
`export PYTHONPATH=$PYTHONPATH:/path-to-folder/.../generalized_latent_equilibrium/`

Restart shell and activate environment again.

## Instructions for running a single MNIST autoencoder experiment with PAL

Run MNIST autoencoder experiment:
`python experiments/le_layers_mnist_training.py --params experiments/single_example/params.json`

## Instructions for reproducing MNIST autoencoder experiment with PAL (parameter sweep and linear classifier)

This will run the experiment required for Fig. 5 in the manuscript. See paper and supplementary information for network parameters.

Important: Run the single experiment before in order to download MNIST dataset.

Then, use the simple runner script (modify parameters for sweep in `runner.py`):

```
cd experiments/Fig5
python runner.py --algorithm PAL --run
python runner.py --algorithm PAL --linclass
python runner.py --algorithm PAL --gather
```

Instead of PAL algorithm, choose from `BP, FA or PAL`.

- `--run` Will train the model, saving latent activation and model after every epoch.
- `--linclass` Will load the model files and run a linear classifier on the test set.
- `--gather` Will gather all results into a .npy file. Run linclass.ipynb to produce Fig. 5. 
