# Experiments on learning the backward weights using cortical microcircuits

This folder contains all files to reproduce several experiments with microcircuits.

Defintions of the classes `base_model` and `phased_noise_model` are given in `microcircuit.py`.
- `base_model` implements backprop (by setting WPP = BPP.T) and feedback alignment as in arXiv:1810.11393
- `phased_noise_model` implements our algorithm to learn BPP

Tasks currently implemented:
- learning the backwards weights with no teaching signal
- feeding input into a teacher network, recording its output; feeding same input into learning network and learning the forward and backward weights simultaneously.

There are two ways to run the experiments:
- open any of the Jupyter notebooks, where all steps are explained
- run the standalone python script by invoking `runner.py`. See below on usage.

While I have implemented multiprocessing in the standalone script, everything still runs on numpy, so the experiments should be kept fairly small.

# Using the standalone `runner.py`

- clone all files and folders from this folder
- generate a new folder for the experiment: `mkdir run01`
- copy the parameter file template: `cp params.json run01/`
- change any parameter in `params.json`
- run with `python runner.py --params run01/params.json --task bw_only` for bw weights only (no teacher)
   or with `python runner.py --params run01/params.json --task fw_bw` for fw and bw weight learning (with teacher)
- all plots are saved in `run01`, together with a `model.pkl` file of the microcircuit class objects after training

To load and re-plot a saved model, run `python runner.py --params run01/params.json --task bw_only --load run01/model.pkl`.

Some tips:
- currently implemented models are `BP`, `FA` and `DTPDRL`
- time parameters (`dt, dtxi, tausyn, Tpres, taueps, Tbw`) are defined in milliseconds.
- *seeds* in `params.json` is an array of numpy random seeds (not a number of seeds)
- *input_signal* in `params.json` defines the signal fed into teacher and students. Currently implemented options: `step`
- setting *rec_per_steps* to anything below 1/dt (standard: 1000) slows down training and generates large .pkl files
- recording too many variables slows down training significantly

Data is recorded in lists such as `uP_breve_time_series`. Every class object (i.e. every microcircuit model) saves its own time series, which can be called with `mc1.uP_breve_time_series`. Every time series has the index structure `uP_breve_time_series[recorded time step][layer][neuron]` for voltages and rates; weight time series are of the form `WPP_breve_time_series[recorded time step][layer][weight]`.

Keep in mind that the `layer` variable always starts from zero. So e.g. for the interneuron recordings, `uI_time_series[-1][0][1]` returns the voltage of the second neuron in the final (and only) layer of interneurons at the end of training.

To load a saved .pkl-file in an interactive Python session, go to the folder where `runner.py` is located, run `python`, and

```
import src.save_exp
import pickle
with open('run03_teacher/model.pkl', 'rb') as f: input = pickle.load(f)
```
After loading this, `input[0]` represents the teacher model (if it was initiated) and other elements are student networks.
