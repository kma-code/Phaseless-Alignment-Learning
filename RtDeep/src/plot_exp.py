import numpy as np
from microcircuit import *
#import time
import logging
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
plt.rcParams['text.usetex'] = False
plt.rc('font', size=12,family='serif')
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter

logging.basicConfig(format='Train model -- %(levelname)s: %(message)s',
                    level=logging.INFO)

def plot(MC_list, path=None):
	if path is not None:
		PATH = path + '/'
	else:
		PATH = ''

	# define the number of time steps which belong to the pre-training
	# and therefore should be skipped in plotting
	TPRE = int(MC_list[0].Tpres / MC_list[0].dt)

	# colors for plotting
	color = cm.rainbow(np.linspace(0, 1, len(MC_list)))

	if MC_list[0].rec_MSE:

		fig = plt.figure()
		for mc, c in zip(MC_list, color):
			plt.plot(mc.MSE_time_series[TPRE:], c=c)

		plt.title("MSE")
		plt.yscale('log')
		# plt.legend()
		plt.savefig(PATH + 'MSE.png', dpi=200)


	if MC_list[0].rec_WPP:

		for i in range(len(MC_list[0].WPP)):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.WPP_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.WPP_time_series[TPRE:]]), c=c)
			plt.title("$W^\\mathrm{PP}$ layer " + str(i+1))
			# plt.grid()
			# plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'WPP_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)

	if MC_list[0].rec_WIP:

		for i in range(len(MC_list[0].WIP)):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.WIP_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.WIP_time_series[TPRE:]]), c=c)
			plt.title("$W^\\mathrm{IP}$ layer " + str(i+1))
			# plt.grid()
			# plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'WIP_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)

	if MC_list[0].rec_BPP:

		for i in range(len(MC_list[0].BPP)):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.BPP_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.BPP_time_series[TPRE:]]), c=c)
			plt.title("$B^\\mathrm{PP}$ layer " + str(i+1))
			# plt.grid()
			# plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'BPP_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)

	if MC_list[0].rec_BPI:

		for i in range(len(MC_list[0].BPI)):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.BPI_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.BPI_time_series[TPRE:]]), c=c)
			plt.title("$B^\\mathrm{PI}$ layer " + str(i+1))
			# plt.grid()
			# plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'BPI_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)



	if MC_list[0].rec_uP:

		for i in range(len(MC_list[0].layers)-1):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.uP_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.uP_time_series[TPRE:]]), c=c)
			plt.title("$u^\\mathrm{P}$ layer " + str(i+1))
			# plt.grid()
			# plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'uP_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)

	if MC_list[0].rec_rP_breve:

		for i in range(len(MC_list[0].layers)-1):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.rP_breve_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.rP_breve_time_series[TPRE:]]), c=c)
			plt.title("$\\breve{r}^\\mathrm{P}$ layer " + str(i+1))
			# plt.grid()
			# plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'rP_breve_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)


 	# the following three variables are only relevant in the last layer
	if MC_list[0].rec_rP_breve_HI:

		for i in range(-1,0):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.rP_breve_HI_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.rP_breve_HI_time_series[TPRE:]]), c=c)
			plt.title("$\\breve{r}^\\mathrm{P}_\\mathrm{HI}$ layer " + str(i+1))
			# plt.grid()
			# plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'rP_breve_HI_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)

	if MC_list[0].rec_uI:

		for i in range(-1,0):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.uI_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.uI_time_series[TPRE:]]), c=c)
			plt.title("$u^\\mathrm{I}$ layer " + str(i+1))
			# plt.grid()
			# plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'uI_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)

	if MC_list[0].rec_rI_breve:

		for i in range(-1,0):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.rI_breve_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.rI_breve_time_series[TPRE:]]), c=c)
			plt.title("$\\breve{r}^\\mathrm{I}$ layer " + str(i+1))
			# plt.grid()
			# plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'rI_breve_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)

	if MC_list[0].rec_vapi:

		for i in range(len(MC_list[0].layers)-2):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.vapi_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.vapi_time_series[TPRE:]]), c=c)
			plt.title("$v^\\mathrm{api}$ before noise injection, layer " + str(i+1))
			# plt.grid()
			# plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'vapi_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)

	if MC_list[0].rec_vapi_noise:

		for i in range(len(MC_list[0].layers)-2):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.vapi_noise_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.vapi_noise_time_series[TPRE:]]), c=c)
			plt.title("$v^\\mathrm{api}$ after noise injection, layer " + str(i+1))
			# plt.grid()
			# plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'vapi_noise_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)

	if MC_list[0].rec_noise:

		for i in range(len(MC_list[0].layers)-2):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.noise_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.noise_time_series[TPRE:]]), c=c)
			plt.title("injected noise layer " + str(i+1))
			# plt.grid()
			# plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'noise_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)

	if MC_list[0].rec_epsilon:

		for i in range(len(MC_list[0].layers)-2):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				plt.plot([vec[i] for vec in mc.epsilon_time_series[TPRE:]], c=c)
			plt.title("$\\epsilon$ layer " + str(i+1))
			plt.grid()
			plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'epsilon_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200) 


	if MC_list[0].rec_epsilon_LO:

		for i in range(len(MC_list[0].layers)-2):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				plt.plot([vec[i] for vec in mc.epsilon_LO_time_series[TPRE:]], c=c)
			plt.title("$\\epsilon_\\mathrm{LO}$ layer " + str(i+1))
			plt.grid()
			plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'epsilon_LO_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200) 

		for i in range(len(MC_list[0].layers)-2):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				plt.plot([1-2*np.array(vec[i]) for vec in mc.epsilon_LO_time_series[TPRE:]], label="$\\cos(B^\\mathrm{PP}r^P, \\breve{u}_i^P)$", c=c)

			lab = str(MC_list[0].noise_deg) + "$^\\circ$"
			plt.plot([np.cos(MC_list[0].noise_deg * np.pi/180) for vec in MC_list[0].epsilon_LO_time_series[TPRE:]], label=lab, ls='--')

			plt.title("$1 - 2 \\;\\epsilon_\\mathrm{LO} \\sim$ cos, layer " + str(i+1))
			# plt.yscale("log")
			plt.legend()
			plt.grid()
			plt.ylim(-1.1,1.1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'cos_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200) 


