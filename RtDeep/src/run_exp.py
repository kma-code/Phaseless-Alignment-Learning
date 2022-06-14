import numpy as np
from microcircuit import *
import time
import logging

logging.basicConfig(format='Train model -- %(levelname)s: %(message)s',
                    level=logging.INFO)

# takes a microcircuit object and runs it based on the signal given

def run(mc, learn=True, teacher=False):
	t_start = time.time()

	logging.info(f"Seed {mc.seed}: initialising recording")

	mc.init_record(rec_per_steps=mc.rec_per_steps,
		rec_MSE=mc.rec_MSE,
		rec_error=mc.rec_error,
		rec_input=mc.rec_input,
		rec_target=mc.rec_target,
		rec_WPP=mc.rec_WPP,
		rec_WIP=mc.rec_WIP,
		rec_BPP=mc.rec_BPP,
		rec_BPI=mc.rec_BPI,
		rec_dWPP=mc.rec_dWPP,
		rec_dWIP=mc.rec_dWIP,
		rec_dBPP=mc.rec_dBPP,
		rec_dBPI=mc.rec_dBPI,
		rec_uP=mc.rec_uP,
		rec_uI=mc.rec_uI,
		rec_uP_breve=mc.rec_uP_breve,
		rec_uI_breve=mc.rec_uI_breve,
		rec_rI_breve=mc.rec_rI_breve,
		rec_rP_breve=mc.rec_rP_breve,
		rec_rP_breve_HI=mc.rec_rP_breve_HI,
		rec_vbas=mc.rec_vbas,
		rec_vapi=mc.rec_vapi,
		rec_vapi_noise=mc.rec_vapi_noise,
		rec_epsilon=mc.rec_epsilon,
		rec_noise=mc.rec_noise,
		rec_epsilon_LO=mc.rec_epsilon_LO)

	# if the mc is the teacher, record target signal
	if teacher:
		mc.target = []

	logging.info(f"Seed {mc.seed}: running pre-training")
	mc = pre_training(mc, r0_arr=mc.input, time=mc.settling_time/mc.dt)

	logging.info(f"Seed {mc.seed}: running training")
	mc = training(mc, r0_arr=mc.input, epochs=mc.epochs, learn=learn, teacher=teacher)

	t_diff = time.time() - t_start
	logging.info(f"Seed {mc.seed}: done in {t_diff}s.")

	return mc


def pre_training(mc, r0_arr, time=None):

	if time == None:
		time = mc.settling_time / mc.dt
	# pre-training to settle voltages -- if we don't do this, weights learn incorrectly due to the incorrect voltages in the beginning
	for i in range(int(time)):
		mc.evolve_system(r0=r0_arr[i], learn_weights=False, learn_bw_weights=False)

	return mc

def training(mc, r0_arr, epochs=1, learn=True, teacher=False):

	for n in range(epochs):
		logging.info(f"Seed {mc.seed}: working on epoch {n}")

		if mc.input_signal == 'step':

			if not teacher and hasattr(mc, 'target'):
				logging.debug(f"Shuffling input and output pairs")
				# # extract unique samples, i.e. first sample after Tpres
				r0_arr = r0_arr[::int(mc.Tpres / mc.dt)]
				target = np.array(mc.target)[int(mc.Tpres / mc.dt)-1:][::int(mc.Tpres / mc.dt)]
				# combine with target voltages
				# new array has axes [time, neuron id, r0/target]
				input_output_arr = np.transpose(np.array([r0_arr, target]), (1,2,0))
				# # shuffle along time axis
				mc.rng.shuffle(input_output_arr)
				# # repeat samples for Tpres times
				input_output_arr = np.repeat(input_output_arr, int(mc.Tpres / mc.dt), axis=0)
				# extract input, ouput
				r0_arr, target = np.transpose(input_output_arr, (2, 0, 1))

				mc.target = target.copy()

			elif not teacher:
				logging.debug(f"Shuffling input")
				# # extract unique samples, i.e. first sample after Tpres
				r0_arr = r0_arr[::int(mc.Tpres / mc.dt)]
				# # shuffle along time axis
				mc.rng.shuffle(r0_arr)
				# # repeat samples for Tpres times
				r0_arr = np.repeat(r0_arr, int(mc.Tpres / mc.dt), axis=0)
			
			

		for i in range(len(r0_arr)):
			# if mc is teacher, evolve and record
			if teacher:
				mc.target.append(copy.deepcopy(mc.uP_breve[-1]))
				mc.evolve_system(r0=r0_arr[i], learn_weights=learn, learn_bw_weights=learn)
			# if target has been defined, use that
			elif hasattr(mc, 'target'):
				mc.evolve_system(r0=r0_arr[i], u_tgt=[mc.target[i]], learn_weights=learn, learn_bw_weights=learn)
			else:
				mc.evolve_system(r0=r0_arr[i], learn_weights=learn, learn_bw_weights=learn)
	return mc



