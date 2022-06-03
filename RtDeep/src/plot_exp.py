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

def plot(MC_list, MC_teacher=None, path=None):

	# define a variable for path of output files
	if path is not None:
		PATH = path + '/'
	else:
		PATH = ''

	# define the number of recorded time steps which belong to the pre-training
	# and therefore should be skipped in plotting
	TPRE = int(MC_list[0].settling_time / MC_list[0].dt / MC_list[0].rec_per_steps)

	# colors for plotting
	color = cm.rainbow(np.linspace(0, 1, len(MC_list)))
	# color of teacher
	CLR_TEACH = 'k'

	if MC_list[0].rec_MSE and MC_teacher is not None:

		fig = plt.figure()
		for mc, c in zip(MC_list, color):
			plt.plot(moving_average(mc.MSE_time_series, int(10*mc.Tpres/mc.dt/mc.rec_per_steps)), c=c)
			# plt.plot(mc.MSE_time_series, c=c)

		plt.title("MSE (window over $10\\;T_\\mathrm{pres}$)")
		plt.yscale('log')
		# plt.legend()
		plt.savefig(PATH + 'MSE.png', dpi=200)


	if MC_list[0].rec_WPP:

		for i in range(len(MC_list[0].WPP)):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.WPP_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.WPP_time_series[TPRE:]]), c=c)
					if MC_teacher is not None:
						plt.plot(np.array([MC_teacher[0].WPP[i][j] for vec in MC_list[0].WPP_time_series[TPRE:]]), c=CLR_TEACH, ls='--')
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
					if MC_teacher is not None:
						plt.plot(np.array([MC_teacher[0].WIP[i][j] for vec in MC_list[0].WIP_time_series[TPRE:]]), c=CLR_TEACH, ls='--')
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
					# if MC_teacher is not None:
					# 	plt.plot(np.array([MC_teacher[0].BPP[i][j] for vec in MC_list[0].BPP_time_series[TPRE:]]), c=CLR_TEACH, ls='--')
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
					# if MC_teacher is not None:
					# 	plt.plot(np.array([MC_teacher[0].BPI[i][j] for vec in MC_list[0].BPI_time_series[TPRE:]]), c=CLR_TEACH, ls='--')
			plt.title("$B^\\mathrm{PI}$ layer " + str(i+1))
			# plt.grid()
			# plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'BPI_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)

	if MC_list[0].rec_BPI and MC_list[0].rec_BPP:

		for i in range(len(MC_list[0].BPI)):
				fig = plt.figure()
				for mc, c in zip(MC_list, color):
					for j in range(len(mc.BPI_time_series[0][i])):
						vec1 = np.array([vec[i][j] for vec in mc.BPI_time_series[TPRE:]])
						vec2 = np.array([vec[i][j] for vec in mc.BPP_time_series[TPRE:]])
						plt.plot(vec1+vec2, c=c)
						# if MC_teacher is not None:
						# 	plt.plot(np.array([MC_teacher[0].BPI[i][j] for vec in MC_list[0].BPI_time_series[TPRE:]]), c=CLR_TEACH, ls='--')
				plt.title("$B^\\mathrm{PI}$ layer " + str(i+1))
				# plt.grid()
				# plt.ylim(0,1)
				plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
				file_name = 'BPP+BPI_layer'+str(i+1)+'.png'
				plt.savefig(PATH + file_name, dpi=200)



	if MC_list[0].rec_uP:

		for i in range(len(MC_list[0].layers)-1):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.uP_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.uP_time_series[TPRE:]]), c=c)
					if MC_teacher is not None:
						data = np.array([vec[i][j] for vec in MC_teacher[0].uP_time_series[TPRE:]])
						data = np.tile(data, MC_list[0].epochs)
						plt.plot(data, c=CLR_TEACH, ls='--')
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
					if MC_teacher is not None:
						data = np.array([vec[i][j] for vec in MC_teacher[0].rP_breve_time_series[TPRE:]])
						data = np.tile(data, MC_list[0].epochs)
						plt.plot(data, c=CLR_TEACH, ls='--')
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
					if MC_teacher is not None:
						data = np.array([vec[i][j] for vec in MC_teacher[0].uI_time_series[TPRE:]])
						data = np.tile(data, MC_list[0].epochs)
						plt.plot(data, c=CLR_TEACH, ls='--')
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
					if MC_teacher is not None:
						data = np.array([vec[i][j] for vec in MC_teacher[0].rI_breve_time_series[TPRE:]])
						data = np.tile(data, MC_list[0].epochs)
						plt.plot(data, c=CLR_TEACH, ls='--')
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
					if MC_teacher is not None:
						data = np.array([vec[i][j] for vec in MC_teacher[0].vapi_time_series[TPRE:]])
						data = np.tile(data, MC_list[0].epochs)
						plt.plot(data, c=CLR_TEACH, ls='--')
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

	if hasattr(MC_list[0], "angle_dWPP_time_series"):

		for i in range(len(MC_list[0].angle_dWPP_time_series[0])):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				plt.plot([vec[i] for vec in mc.angle_dWPP_time_series], c=c)
			plt.title("$\\angle (\\Delta W^\\mathrm{PP}, \\mathrm{BP})$ layer " + str(i+1))
			plt.ylabel("deg")
			plt.grid()
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'angle_BP_dWPP'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200) 

	if hasattr(MC_list[0], "angle_WPPT_BPP_time_series"):

		for i in range(len(MC_list[0].angle_WPPT_BPP_time_series[0])):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				plt.plot([vec[i] for vec in mc.angle_WPPT_BPP_time_series[TPRE:]], c=c)
			plt.title("$\\angle (B^\\mathrm{PP}, (W^\\mathrm{PP})^T )$ layer " + str(i+1))
			plt.ylabel("deg")
			plt.grid()
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'angle_WPPT_BPP'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200) 

			# create a mean + std plot
			fig = plt.figure()
			data = []
			for mc, c in zip(MC_list, color):
				data.append([vec[i] for vec in mc.angle_WPPT_BPP_time_series[TPRE:]])
			mean = np.mean(data, axis=0)
			std = np.std(data, axis=0)
			x = np.arange(len(mean))

			plt.plot(x, mean, c='k')
			plt.title("$\\angle (B^\\mathrm{PP},(W^\\mathrm{PP})^T)$ layer " + str(i+1))
			plt.ylabel("deg")
			plt.fill_between(x, mean+std, mean-std, color='gray', alpha=.5)
			plt.grid()
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'angle_WPPT_BPP'+str(i+1)+'_mean.png'
			plt.savefig(PATH + file_name, dpi=200) 

	if hasattr(MC_list[0], "angle_jacobians_BP_time_series"):

		for i in range(len(MC_list[0].angle_jacobians_BP_time_series[0])):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				plt.plot([vec[i] for vec in mc.angle_jacobians_BP_time_series[TPRE:]], c=c)
			plt.title("$\\angle (J_g, J_f^T)$ layer " + str(i+1))
			plt.ylabel("deg")
			plt.grid()
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'angle_jacobians_BP'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200) 

			# create a mean + std plot
			fig = plt.figure()
			data = []
			for mc, c in zip(MC_list, color):
				data.append([vec[i] for vec in mc.angle_jacobians_BP_time_series[TPRE:]])
			mean = np.mean(data, axis=0)
			std = np.std(data, axis=0)
			x = np.arange(len(mean))

			plt.plot(x, mean, c='k')
			plt.title("$\\angle (J_g, J_f^T)$ layer " + str(i+1))
			plt.ylabel("deg")
			plt.fill_between(x, mean+std, mean-std, color='gray', alpha=.5)
			plt.grid()
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'angle_jacobians_BP'+str(i+1)+'_mean.png'
			plt.savefig(PATH + file_name, dpi=200) 

	if hasattr(MC_list[0], "angle_BPP_RHS_time_series"):

		for i in range(len(MC_list[0].angle_BPP_RHS_time_series[0])):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				plt.plot([vec[i] for vec in mc.angle_BPP_RHS_time_series[TPRE:]], c=c)
			plt.title("$\\angle (B^\\mathrm{PP}, \\varphi' (W^\\mathrm{PP})^T \\varphi')$ layer " + str(i+1))
			plt.ylabel("deg")
			plt.grid()
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'angle_BPP_RHS'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200) 

			# create a mean + std plot
			fig = plt.figure()
			data = []
			for mc, c in zip(MC_list, color):
				data.append([vec[i] for vec in mc.angle_BPP_RHS_time_series[TPRE:]])
			mean = np.mean(data, axis=0)
			std = np.std(data, axis=0)
			x = np.arange(len(mean))

			plt.plot(x, mean, c='k')
			plt.title("$\\angle (B^\\mathrm{PP}, \\varphi' (W^\\mathrm{PP})^T \\varphi')$ layer " + str(i+1))
			plt.ylabel("deg")
			plt.fill_between(x, mean+std, mean-std, color='gray', alpha=.5)
			plt.grid()
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'angle_BPP_RHS'+str(i+1)+'_mean.png'
			plt.savefig(PATH + file_name, dpi=200) 

