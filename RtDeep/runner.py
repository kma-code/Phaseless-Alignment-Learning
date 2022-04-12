# Loads a paramater file and executes microcircuit.py

# %matplotlib widget

import numpy as np
import matplotlib.pyplot as plt

from microcircuit import *
import src.init_MC as init_MC
import src.init_signals as init_signals
import src.run_exp as run_exp
import src.plot_exp as plot_exp
import src.save_exp as save_exp

import sys
import os
import argparse
import logging
import json
import multiprocess as mp
import time

N_PROCESSES = 12 # defined by compute setup


logging.basicConfig(format='Train model -- %(levelname)s: %(message)s',
                    level=logging.INFO)

def parse_experiment_arguments():
	"""
		Parse the arguments for the test and train experiments
	"""

	parser = argparse.ArgumentParser(description='Train a model on the mnist \
		dataset.')
	parser.add_argument('--params', type=str,
						help='Path to the parameter .py-file.')
	parser.add_argument('--task', type=str,
						help='Choose from <bw_only> or <fw_bw>.',
						default='bw-only')
	parser.add_argument('--load', type=str,
						help='Load a previously saved model from a .pkl file.',
						default=None)
	args = parser.parse_args()

	return args


def main(params, task='fw_bw', seeds=[667], target=None, load=None):



	t_start = time.time()

	if load is None:

		MC_list = init_MC.init_MC(params, seeds)
		logging.info(f'List of initialised MCs: {MC_list}')

		# init input signal
		MC_list = init_signals.init_r0(MC_list=MC_list, form=params["input_signal"])

		logging.debug(f"Current state of networks:")
		for mc in MC_list:
			logging.debug(f"Voltages: {mc.uP}, {mc.uI}")
			logging.debug(f"Weights: {mc.WPP}, {mc.WIP}, {mc.BPP}, {mc.BPI}")
			logging.debug(f"Input: {mc.input}")

		logging.info(f'Model: {params["model_type"]}')
		logging.info(f'Task: {task}')
		logging.info(f'Target: {target}')

		# setup multiprocessing
		# one process for every initialised mc
		if N_PROCESSES == 1:
			logging.info(f'Number of processes set to {N_PROCESSES}. Why? I mean, you could use 4. Or 16. But whatever, I\'ll go ahead.')

		logging.info(f'Setting up and running {N_PROCESSES} processes')

		if N_PROCESSES == 1:
			logging.info(f'This is going to take an awfully long time because user defined {N_PROCESSES} processes.')

		with mp.Pool(N_PROCESSES) as pool:
			MC_list = pool.map(run_exp.run, MC_list)
			pool.close()

		t_diff = time.time() - t_start
		logging.info(f'Training finished in {t_diff}s.')

		logging.info(f'Saving results')
		save_exp.save(MC_list, path=PATH)

	else:

		logging.info(f'Loading results from {load}')
		MC_list = save_exp.load(load)



	logging.info(f'Plotting results')
	# plot.
	plot_exp.plot(MC_list, path=PATH)


	t_diff = time.time() - t_start
	logging.info(f"Done. Total time: {t_diff}s")






if __name__ == '__main__':

	ARGS = parse_experiment_arguments()

	# get path of parameter file
	PATH = os.path.dirname(ARGS.params)
	if PATH == '':
		PATH = None

	# logging.info('Importing parameters')
	with open(ARGS.params, 'r+') as f:
		PARAMETERS = json.load(f)
	logging.info('Sucessfully imported parameters')
	logging.debug(PARAMETERS)

	main(params=PARAMETERS, task=ARGS.task, seeds=PARAMETERS['random_seed'], load=ARGS.load)



