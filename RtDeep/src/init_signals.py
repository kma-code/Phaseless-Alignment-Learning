import numpy as np
from microcircuit import *

def init_r0(MC_list, form="step", seed=123):

	rng = np.random.RandomState(seed)

	if form == "step":
		# input rates: step functions in the form of random inputs held for Tpres
		r0 = rng.uniform(0, 1, size=(MC_list[0].dataset_size, MC_list[0].layers[0]))
		r0 = np.repeat(r0, int(MC_list[0].Tpres / MC_list[0].dt), axis=0)

	for mc in MC_list:
		mc.input = r0

	return MC_list