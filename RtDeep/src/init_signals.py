import numpy as np
from microcircuit import *

def init_r0(MC_list, form="step"):

	for mc in MC_list:
		np.random.seed(mc.seed)

		if form == "step":
			# input rates: step functions in the form of random inputs held for Tpres
			r0 = np.random.uniform(0, 1, size=(mc.dataset_size, mc.layers[0]))
			r0 = np.repeat(r0, int(mc.Tpres / mc.dt), axis=0)

		mc.input = r0

	return MC_list