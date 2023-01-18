# Experiments on learning the backward weights using cortical microcircuits

This folder contains all files to reproduce several experiments with microcircuits.

Defintions of the classes `base_model` and `noise_model` are given in `microcircuit.py`.
- `base_model` implements backprop (by setting WPP = BPP.T) and feedback alignment as in arXiv:1810.11393
- `noise_model` implements our algorithm PAL to learn BPP

Parameters:

Tasks currently implemented:
- 'bw_only': learning the backwards weights with no teaching signal
- 'fw_bw': feeding input into a teacher network, recording its output; feeding same input into learning network and learning the forward and backward weights simultaneously.

Input signal:
- Steps sampled from U[0,1] held for Tpres

There are two ways to run the experiments:
- open any of the Jupyter notebooks, where all steps are explained
- run the standalone python script by invoking `runner.py`. See below on usage.

While I have implemented multiprocessing in the standalone script, everything still runs on numpy, so the experiments should be kept fairly small.

# Using the standalone `runner.py`

You can either make an experiment using the template:

- clone all files and folders from this folder
- generate a new folder for the experiment: `mkdir run01`
- copy the parameter file template: `cp params.json run01/`
- change any parameter in `params.json`
- run with `python runner.py --params run01/params.json --task bw_only` for bw weights only (no teacher)
   or with `python runner.py --params run01/params.json --task fw_bw` for fw and bw weight learning (with teacher)
- if
- all plots are saved in `run01`, together with a `model.pkl` file of the microcircuit class objects after training

Or use one of the files for the figures in the plot. Go through the above steps with one of the params.json files in the folder [experiments](https://github.com/kma-code/Phaseless-Alignment-Learning/tree/master/PAL%20MC/experiments).

Models are saved after training and once again after plotting.
To load and re-plot a saved model, run `python runner.py --params .../params.json --task bw_only --load .../model.pkl`.

Some tips:
- implemented models are `BP`, `FA` and `PAL`
- time parameters (`dt, dtxi, tauHP, tauLO, Tpres`) are defined in milliseconds. Their meaning is explained above.
- *seeds* in `params.json` is an array of numpy random seeds (not a number of seeds)
- *input_signal* in `params.json` defines the signal fed into teacher and students. Currently implemented options: `step`
- setting *rec_per_steps* to anything below 1/dt (standard: 1000) slows down training and generates large .pkl files
- recording too many variables slows down training significantly

Data is recorded in lists such as `uP_breve_time_series`. Every class object (i.e. every microcircuit model) saves its own time series, which can be called with e.g. `mc1.uP_breve_time_series`. Every time series has the index structure `uP_breve_time_series[recorded time step][layer][neuron index]` for voltages and rates; weight time series are of the form `WPP_breve_time_series[recorded time step][layer][weight index]`.

Keep in mind that the `layer` variable always starts from zero. So e.g. for the interneuron recordings, `uI_time_series[-1][0][1]` returns the voltage of the second neuron in the final (and only) layer of interneurons at the end of training.

To load a saved .pkl-file in an interactive Python session, go to the folder where `runner.py` is located, run `python`, and

```
import src.save_exp
import pickle
with open('run03_teacher/model.pkl', 'rb') as f: input = pickle.load(f)
```
After loading this, `input[0]` represents the teacher model (if it was initiated) and other elements are student networks.

# Commands to reproduce plots

- Fig3abc: `python runner.py --params experiments/Fig3abc/params.json --task bw_only --compare BP`
- Fig3def: `python runner.py --params experiments/Fig3def/params.json --task bw_only --compare BP`
- Fig4ab: `python runner.py --params experiments/Fig4ab/PAL/params.json --task fw_bw --compare BP`

