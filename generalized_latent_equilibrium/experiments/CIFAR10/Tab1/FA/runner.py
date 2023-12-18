import argparse
import numpy as np
import json
import subprocess
from pathlib import Path
import os
import pickle
import logging

if __name__ == '__main__':

		logging.basicConfig(format='Generating param files -- %(levelname)s: %(message)s', level=logging.INFO, force=True)

		parser = argparse.ArgumentParser(description='Train a Latent Equilibrium Neural Network on CIFAR10 task.')
		parser.add_argument('--run', action='store_true',
														default=False,
														help='Run parameter sweep.')
		parser.add_argument('--gather', action='store_true',
														default=False,
														help='Gather results.')
		parser.add_argument('--algorithm', required=True,
														help='Choose algorithm: BP, FA or PAL')


		args = parser.parse_args()
		algo = args.algorithm
		logging.info(f'Algorithm: {algo}')

		# path to parent generalized_latent_equilbrium folder
		PATH_parent = Path(__file__).parent.resolve().parents[2]
		# path to folder of this file
		PATH_runner = Path(__file__).parent.resolve() / str(algo)



		OUTPUT_DIR = PATH_runner / "runs"

		# create a bunch of JSON param files in subfolders (lr/seeds/)
		seeds = [1,3,5,7,9,11,13,15,17,19]

		params_arr = []
		for lr in [None]:
			params_per_lr = []
			for seed in seeds:
				params = {
						"algorithm": algo,
						"epochs": 50,
						"batch_size": 128,
						"batch_learning_multiplier": 64,
						"lr_factors": [10.0, 0.0, 25.0, 0.0, 2.0, 0.2],
						#"lr_factors": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
						"wn_sigma": [0,0,0,0,0,0],
						"n_updates": 100,
						"target_type": "mse",
						"seed": seed,
						"activation": "Sigmoid",
						"model_variant": "vanilla",
						"with_optimizer": "store_true",
						"rec_degs": True,						 
						}
				if algo == 'PAL':
						params.update(
												{
								"bw_lr_factors": [0,0,0,0,10,10],
								"regularizer": [0,0,0,0,5e-5,5e-5],
								"tau_xi": [10,10,10,10,10,10],
								"tau_HP": [10,10,10,10,10,10],
								"tau_LO": [0,0,0,0,0,0],
								"sigma": [0,0,0,5e-2,5e-2,0]
								}
						)
				params_per_lr.append(params)
		params_arr.append(params_per_lr)



		if args.run and not args.gather:
				for i, params_per_lr in zip(range(len(params_arr)), params_arr):
						for j, params in zip(range(len(params_per_lr)), params_per_lr):

								PATH_output = OUTPUT_DIR / str('lr' + str(i) +	'/seed' + str(j))
								params["output"] = str(PATH_output) + '/'

								# create output directory if it doesn't exist
								if not(os.path.exists(params['output'])):
												# logging.info(f"{PATH_runner + '/' + params['output'] } doesn't exists, creating")
												os.makedirs(params['output'] )

								with open(str(params['output']) + '/params.json', 'w') as f:
												logging.info(f"Saving to {f.name}")
												json.dump(params, f)

								sim_dir = PATH_output
								# start runs as separate processes
								proc_name = ['sbatch', 'slurm_script_gpu_preempt.sh', 'python', 'experiments/CIFAR10/le_layers_cifar10_training.py', '--params', str(sim_dir / 'params.json')]

								logging.info(f"Starting run as subprocess {proc_name}.")
								subprocess.Popen(proc_name, cwd=PATH_parent)

		elif args.gather:
				# Collect results from linear classifiers
				lin_acc_arr = []
				for i, params_per_lr in zip(range(len(params_arr)), params_arr):
						lin_acc_per_lr = []
						for j, params in zip(range(len(params_per_lr)), params_per_lr):

								sim_dir = OUTPUT_DIR / str('lr' + str(i) +		'/seed' + str(j))
								with open(str(sim_dir) + '/val_acc.pkl', 'rb') as in_file:
										val_acc = pickle.load(in_file)
								lin_acc_per_lr.append(val_acc)
						lin_acc_arr.append(lin_acc_per_lr)


				lin_acc_output_file = PATH_runner / "lin_acc_lr_seeds_epochs.npy"
				np.save(lin_acc_output_file, lin_acc_arr)
				logging.info(f"Gathered data and saved to {lin_acc_output_file}.")




