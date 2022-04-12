import numpy as np
from microcircuit import *
import time
import logging

logging.basicConfig(format='Train model -- %(levelname)s: %(message)s',
                    level=logging.INFO)

# takes a microcircuit object and runs it based on the signal given

def run(mc):
	t_start = time.time()

	logging.info(f"Seed {mc.seed}: initialising recording")

	mc.init_record(rec_per_steps=mc.rec_per_steps,
		rec_MSE=mc.rec_MSE,
		rec_WPP=mc.rec_WPP,
		rec_WIP=mc.rec_WIP,
		rec_BPP=mc.rec_BPP,
		rec_BPI=mc.rec_BPI,
		rec_uP=mc.rec_rI_breve,
		rec_uI=mc.rec_rI_breve,
		rec_rI_breve=mc.rec_rI_breve,
		rec_rP_breve=mc.rec_rP_breve,
		rec_rP_breve_HI=mc.rec_rP_breve_HI,
		rec_vapi=mc.rec_vapi,
		rec_vapi_noise=mc.rec_vapi_noise,
		rec_epsilon=mc.rec_epsilon,
		rec_noise=mc.rec_noise,
		rec_epsilon_LO=mc.rec_epsilon_LO)

	logging.info(f"Seed {mc.seed}: running pre-training")
	pre_training(mc, r0=mc.input, time=mc.Tpres/mc.dt)

	logging.info(f"Seed {mc.seed}: running training")
	training(mc, r0=mc.input, epochs=mc.epochs)

	t_diff = time.time() - t_start
	logging.info(f"Seed {mc.seed}: done in {t_diff}s.")

	return mc


def pre_training(mc, r0, time=None):

	if time == None:
		time = mc.Tpres / mc.dt
	# pre-training to settle voltages -- if we don't do this, weights learn incorrectly due to the incorrect voltages in the beginning
	for i in range(int(time)):
		mc.evolve_system(r0=r0[i], learn_weights=False, learn_bw_weights=False)

def training(mc, r0, epochs=1):

	for n in range(epochs):
		logging.info(f"Seed {mc.seed}: working on epoch {n}")
		for data in r0:
			mc.evolve_system(r0=data, learn_weights=True, learn_bw_weights=True)



