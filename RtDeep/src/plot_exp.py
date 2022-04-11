import numpy as np
from microcircuit import *
#import time
import logging
import pylab as plt
plt.rc('text', usetex=True)
plt.rc('font', size=12,family='serif')
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter

logging.basicConfig(format='Train model -- %(levelname)s: %(message)s',
                    level=logging.INFO)

def plot(MC_list):
	if MC_list[0].rec_MSE:

		fig = plt.figure()
		for mc in MC_list:
			plt.plot(mc.MSE_time_series)

		plt.title("MSE")
		plt.yscale('log')
		# plt.legend()
		plt.savefig('MSE.png', dpi=200) 


	if MC_list[0].rec_epsilon_LO:

		fig = plt.figure()
		for mc in MC_list:
			plt.plot(np.array(mc.epsilon_LO_time_series).ravel())
		plt.title("$\\epsilon_\\mathrm{LO}$ layer 1")
		plt.grid()
		plt.ylim(0,1)
		plt.savefig('epsilon_LO.png', dpi=200) 

		fig = plt.figure()
		for mc in MC_list:
			plt.plot(1-2*np.array(mc.epsilon_LO_time_series).ravel(), label="$\\cos(B^\\mathrm{PP}r^P, \\breve{u}_i^P)$")

		lab = str(MC_list[0].noise_deg) + "$^\\circ$"
		plt.plot([np.cos(MC_list[0].noise_deg * np.pi/180) for vec in np.array(MC_list[0].epsilon_LO_time_series).ravel()], label=lab, ls='--')

		plt.title("$1 - 2 \\;\\epsilon_\\mathrm{LO} \\sim$ cos, layer 1")
		# plt.yscale("log")
		plt.legend()
		plt.grid()
		plt.ylim(-1.1,1.1)
		plt.savefig('cos.png', dpi=200) 


