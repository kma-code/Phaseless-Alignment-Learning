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

## Instructions for running a single MNIST autoencoder experiment with PAL

Run MNIST autoencoder experiment:
`python experiments/le_layers_mnist_training.py --params experiments/single_example/params.json`

## Instructions for reproducing MNIST autoencoder experiment with PAL (parameter sweep and linear classifier)

Important: Run the single experiment before in order to download MNIST dataset.

Then, use the simple runner script (modify parameters for sweep in `runner.py`):

```
cd experiments/
python runner.py --algorithm PAL --run
python runner.py --algorithm PAL --linclass
python runner.py --algorithm PAL --gather
```

Instead of algorithms, choose from `BP, FA or PAL`.

- `--run` Will train the model, 
