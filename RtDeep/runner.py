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

N_MAX_PROCESSES = 12 # defined by compute setup


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
						default='fw_bw')
	parser.add_argument('--load', type=str,
						help='Load a previously saved model from a .pkl file.',
						default=None)
	args = parser.parse_args()

	return args


def main(params, task='fw_bw', seeds=[667], load=None):



	t_start = time.time()

	if load is None:

		MC_list = init_MC.init_MC(params, seeds)
		logging.info(f'List of initialised student MCs: {MC_list}')

		if task == 'fw_bw':
			MC_teacher = init_MC.init_MC(params, seeds, teacher=True)
			MC_teacher[0].set_self_predicting_state()
			logging.info(f'Teacher initialised with seed {MC_teacher[0].seed}')

			MC_teacher = init_signals.init_r0(MC_list=MC_teacher, form=params["input_signal"], seed=MC_teacher[0].seed)

			for mc in MC_list:
				if mc.copy_teacher_weights:
						logging.info(f'Copying teacher weights')
						mc.set_weights(model=MC_teacher[0])
				if mc.copy_teacher_voltages:
					logging.info(f'Copying teacher voltages')
					mc.set_voltages(model=MC_teacher[0])

		# init input signal
		# if no teacher is present, use seed of first microcircuit
		if task == 'bw_only':
			MC_list = init_signals.init_r0(MC_list=MC_list, form=params["input_signal"], seed=MC_list[0].seed)
		else:
			MC_list = init_signals.init_r0(MC_list=MC_list, form=params["input_signal"], seed=MC_teacher[0].seed)


		logging.info(f'Model: {params["model_type"]}')
		logging.info(f'Task: {task}')

		if task == 'fw_bw':
			logging.info(f'Running teacher to obtain target signal')

			MC_teacher = [run_exp.run(MC_teacher[0], learn=False, teacher=True)]

			logging.info(f'Setting uP_breve in output layer of teacher as target')

			for mc in MC_list:
				# use uP_breve time series after settling time
				mc.target = MC_teacher[0].target

		logging.debug(f"Current state of networks:")
		for mc in MC_list:
			logging.debug(f"Voltages: {mc.uP}, {mc.uI}")
			logging.debug(f"Weights: {mc.WPP}, {mc.WIP}, {mc.BPP}, {mc.BPI}")
			logging.debug(f"Input: {mc.input}")

		N_PROCESSES = len(MC_list) if N_MAX_PROCESSES > len(MC_list) else N_MAX_PROCESSES

		logging.info(f'Setting up and running {N_PROCESSES} processes')

		with mp.Pool(N_PROCESSES) as pool:
			MC_list = pool.map(run_exp.run, MC_list)
			pool.close()

		t_diff = time.time() - t_start
		logging.info(f'Training finished in {t_diff}s.')

		logging.info(f'Saving results')
		if task == 'fw_bw':
			# if teacher is loaded, append to list of microcircuits
			save_exp.save(MC_teacher + MC_list,path=PATH)
		else:
			save_exp.save(MC_list, path=PATH)

	else:
		logging.info(f'Loading results from {load}')
		MC_list = save_exp.load(load)
		if task == 'fw_bw':
			MC_teacher, MC_list = MC_list[0], MC_list[1:]




	logging.info(f'Plotting results')
	# plot.
	if task == 'fw_bw':
		plot_exp.plot(MC_list, MC_teacher, path=PATH)
	else:
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

	if ARGS.task not in ['bw_only', 'fw_bw']:
		raise ValueError("Task not recognized. Use 'bw_only' or 'fw_bw'")

	main(params=PARAMETERS, task=ARGS.task, seeds=PARAMETERS['random_seed'], load=ARGS.load)



