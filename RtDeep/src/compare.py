import numpy as np
from microcircuit import *
import src.init_MC as init_MC
import src.init_signals as init_signals
import src.run_exp as run_exp
import logging

logging.basicConfig(format='Train model -- %(levelname)s: %(message)s',
                    level=logging.INFO)

# define mappings from s
derivative_mappings = {
	linear: d_linear,
    relu: d_relu,
    soft_relu: d_soft_relu,
    logistic: d_logistic,
    tanh: d_tanh,
   	hard_sigmoid: d_hard_sigmoid
}


def calc_dWPP_ANN(mc, W_list, activation_list, d_activation_list, r0, target):
	'''

		Calculates the weight updates in an ANN using BP
		for given input and weights.

		Input:  mc: microcircuit (needed for parameters)
				W_list: list of weights for ANN
				r0: input vector
				target: target vector
		Returns: list of updates dWPP in ANN

	'''

	# forward pass
	# voltages correspond to vbashat
	voltages = [np.zeros_like(vec) for vec in mc.uP_breve]
	rates = [np.zeros_like(vec) for vec in mc.rP_breve]
	voltages[0] = W_list[0] @ r0

	for i in range(len(W_list)-1):
		rates[i] = activation_list[i](voltages[i])
		voltages[i+1] = mc.gbas / (mc.gbas + mc.gapi + mc.gl) * W_list[i+1] @ rates[i]
	# correct output layer voltage
	voltages[-1] = (mc.gbas + mc.gapi + mc.gl) / (mc.gbas + mc.gl) * voltages[-1]
	rates[-1] = activation_list[-1](voltages[-1])

	# backward pass:
	dWPP_BP_list = [np.zeros_like(W) for W in W_list]

	# calculate output error on voltage level at ouput
	#error = np.diag(d_activation_list[-1](voltages[-1])) @ (target - rates[-1])
	error = (target - voltages[-1])
	# propagate error backwards
	for i in range(len(dWPP_BP_list)-1, 0, -1):
		dWPP_BP_list[i] = np.outer(error, rates[i-1])
		error = np.diag(d_activation_list[i-1](voltages[i-1])) @ W_list[i].T @ error
	dWPP_BP_list[0] = np.outer(error, r0)

	return dWPP_BP_list




def compare_updates(mc, model, params):
	"""
		Compares updates of mc (trained microcircuit model)
		given input/output pairs with an ANN with same weights

	"""

	# compare updates of mc object to dWPP of BP

	# define the number of recorded time steps which belong to the pre-training
	# and therefore should be skipped
	TPRE = int(mc.settling_time / mc.dt / mc.rec_per_steps)

	if model == "BP":

		# generate an input/target sequence
		MC_teacher = init_MC.init_MC(params, params['random_seed'], teacher=True)
		MC_teacher[0].set_self_predicting_state()

		logging.info(f'Teacher initialised with seed {MC_teacher[0].seed}')
		# MC_teacher = init_signals.init_r0(MC_list=MC_teacher, form=params["input_signal"], seed=MC_teacher[0].seed)
		# copy input of mc to teacher
		MC_teacher[0].input = mc.input
		
		logging.info(f'Running teacher to obtain target signal')
		MC_teacher = [run_exp.run(MC_teacher[0], learn_weights=False, learn_lat_weights=False, learn_bw_weights=False, teacher=True)]

		logging.info(f'Assigning input/target pairs of teacher to students')
		# mc = init_signals.init_r0(MC_list=[mc], form=params["input_signal"], seed=MC_teacher[0].seed)[0]
		# for mc in MC_list:
		# use uP_breve time series after settling time
		mc.target = MC_teacher[0].target

		# record dWPP for given sequence
		# for mc in MC_list:
		d_activation_list = [derivative_mappings[activation] for activation in mc.activation]

		logging.info(f'Evaluating dWPP for microcircuit {mc}')
		mc.dWPP_time_series_compare = [] 	# dWPP for this mc
		mc.dWPP_time_series_ANN = []		# dWPP for ANN with forward weights of this mc
		mc.angle_updates_time_series = []	# angle between the entries of the above two lists

		WPP_num = len(mc.WPP_time_series[TPRE:])	# length of WPP time series for progess counter

		for time, (WPP, WIP, BPP, BPI) in enumerate(zip(mc.WPP_time_series[TPRE:], mc.WIP_time_series[TPRE:], mc.BPP_time_series[TPRE:], mc.BPI_time_series[TPRE:])):
			logging.info(f"Evaluating next set of weights {time}/{WPP_num}")
			# set network to weights at this time step
			mc.set_weights(WPP=WPP, WIP=WIP, BPP=BPP, BPI=BPI)
			# mc.set_self_predicting_state()

			# run with input, output pairs and record dWPP
			for i, (r0, target) in enumerate(zip(mc.input, mc.target)):
				mc.evolve_system(r0=r0, u_tgt=target, learn_weights=False, learn_lat_weights=False, learn_bw_weights=False, record=False)
				# record dWPP after every presentation time
				# if (i+1) % (mc.Tpres / mc.dt) == 0:
				# get weights in MC and ANN
				mc.dWPP_time_series_compare.append(mc.dWPP)
				mc.dWPP_time_series_ANN.append(calc_dWPP_ANN(mc=mc, W_list=mc.WPP, activation_list=mc.activation, d_activation_list=d_activation_list, r0=r0, target=target))

				# calculate angle between weights
				mc.angle_updates_time_series.append([
					deg(cos_sim(mc_dWPP, BP_dWPP)) for mc_dWPP, BP_dWPP in zip(mc.dWPP_time_series_compare[-1], mc.dWPP_time_series_ANN[-1])
					])

		return mc


def compare_updates_bw_only(MC_list, model):
	# compare updates of mc object to dWPP of BP, even if eta_fw = 0
	# this is done by reconstructing the dWPP of our model

	# define the number of recorded time steps which belong to the pre-training
	# and therefore should be skipped
	TPRE = int(MC_list[0].settling_time / MC_list[0].dt / MC_list[0].rec_per_steps)

	if model == "BP":

		for mc in MC_list:
			d_activation_list = [derivative_mappings[activation] for activation in mc.activation]

			mc.angle_dWPP_bw_only_time_series = []

			# for every time step
			for i in range(len(mc.dWPP_time_series[TPRE:])):
				angle_dWPP_bw_only_arr = []

				# for every layer
				for j in range(len(mc.layers)-1):
					# print("j", j)
					if j == 0:
						r_in = r0
					else:
						r_in = mc.rP_breve_time_series[TPRE:][i][j-1]

					# construct weight update for BP net

					# for output layer
					# print(j, len(r_int), len(mc.rP_breve_time_series))
					# print(len(mc.dWPP_time_series[TPRE:]), len(mc.error_time_series))
					dWPP_BP = np.outer(mc.error_time_series[i], r_in)

					# multiply phi' @ W.T from left
					for k in range(len(mc.layers)-2, j, -1):
						# print(j, k)
						dWPP_BP = np.diag(d_activation_list[k-1](mc.uP_breve_time_series[TPRE:][i][k-1])) @ mc.WPP_time_series[TPRE:][i][k].T @ dWPP_BP

					cos = cos_sim(mc.dWPP_time_series[TPRE:][i][j], -dWPP_BP)

					angle_dWPP_bw_only_arr.append(deg(cos))
					# print(angle_dWPP_arr[-1])

				mc.angle_dWPP_bw_only_time_series.append(angle_dWPP_bw_only_arr)

		return MC_list


def compare_weight_matrices(MC_list, model):

	if model == "BP":

		for mc in MC_list:
			# d_activation_list = [derivative_mappings[activation] for activation in mc.activation]

			mc.angle_WPPT_BPP_time_series = []

			# for every time step
			for i in range(len(mc.BPP_time_series)):
				angle_WPPT_BPP_arr = []

				# for every hidden layer
				for j in range(len(mc.layers)-2):

					if MC_list[0].bw_connection_mode == 'skip':
						# construct Jacobian J_f^T
						WPP_T = np.eye(mc.layers[-1])
						# multiply phi' @ W.T from left
						for k in range(len(mc.layers)-2, j, -1):
							WPP_T = mc.WPP_time_series[i][k].T @ WPP_T

						# construct Jacobian J_g
						BPP = mc.BPP_time_series[i][j]

					elif MC_list[0].bw_connection_mode == 'layered':
						WPP_T = mc.WPP_time_series[i][j+1].T
						BPP   = mc.BPP_time_series[i][j]

					

					cos = cos_sim(WPP_T, BPP)

					angle_WPPT_BPP_arr.append(deg(cos))
					# print(angle_dWPP_arr[-1])

				mc.angle_WPPT_BPP_time_series.append(angle_WPPT_BPP_arr)

		return MC_list


def compare_jacobians(MC_list, model):

	if model == "BP":

		for mc in MC_list:
			d_activation_list = [derivative_mappings[activation] for activation in mc.activation]

			mc.angle_jacobians_BP_time_series = []

			# for every time step
			for i in range(len(mc.BPP_time_series)):
				angle_jacobians_BP_arr = []

				# for every hidden layer
				for j in range(len(mc.layers)-2):

					if MC_list[0].bw_connection_mode == 'skip':
						# construct Jacobian J_f^T
						J_f_T = np.eye(mc.layers[-1])
						# multiply phi' @ W.T from left
						for k in range(len(mc.layers)-2, j, -1):
							J_f_T = np.diag(d_activation_list[k-1](mc.gbas / (mc.gbas + mc.gapi + mc.gl) * mc.vbas_time_series[i][k-1])) @ mc.WPP_time_series[i][k].T @ J_f_T

						# construct Jacobian J_g
						J_g = mc.BPP_time_series[i][j] @ np.diag(d_activation_list[-1](mc.gbas / (mc.gbas + mc.gapi + mc.gl) * mc.vbas_time_series[i][-1]))

					elif MC_list[0].bw_connection_mode == 'layered':
						J_f_T = np.diag(d_activation_list[j](mc.gbas / (mc.gbas + mc.gapi + mc.gl) * mc.vbas_time_series[i][j])) @ mc.WPP_time_series[i][j+1].T
						J_g   = mc.BPP_time_series[i][j] @ np.diag(d_activation_list[j+1](mc.gbas / (mc.gbas + mc.gapi + mc.gl) * mc.vbas_time_series[i][j+1]))

					

					cos = cos_sim(J_g, J_f_T)

					angle_jacobians_BP_arr.append(deg(cos))
					# print(angle_dWPP_arr[-1])

				mc.angle_jacobians_BP_time_series.append(angle_jacobians_BP_arr)

		return MC_list


def compare_BPP_RHS(MC_list, model):

	# similar to compare_jacobians, but involves the 'correct' result of what BPP should converge to
	# in equations, that should be BPP ~ phi' WPP.T phi'

	if model == "BP":

		for mc in MC_list:
			d_activation_list = [derivative_mappings[activation] for activation in mc.activation]

			mc.angle_BPP_RHS_time_series = []

			# for every time step
			for i in range(len(mc.BPP_time_series)):
				angle_BPP_RHS_arr = []

				# for every hidden layer
				for j in range(len(mc.layers)-2):

					if MC_list[0].bw_connection_mode == 'skip':
						# construct Jacobian J_f^T
						J_f_T = np.eye(mc.layers[-1])
						# multiply phi' @ W.T from left
						for k in range(len(mc.layers)-2, j, -1):
							RHS = np.diag(d_activation_list[k-1](mc.gbas / (mc.gbas + mc.gapi + mc.gl) * mc.vbas_time_series[i][k-1])) @ mc.WPP_time_series[i][k].T @ J_f_T

						RHS = RHS @ np.diag(d_activation_list[j+1](mc.gbas / (mc.gbas + mc.gapi + mc.gl) * mc.vbas_time_series[i][j+1]))
						# construct Jacobian J_g
						BPP = mc.BPP_time_series[i][j]

					elif MC_list[0].bw_connection_mode == 'layered':
						RHS = np.diag(d_activation_list[j](mc.gbas / (mc.gbas + mc.gapi + mc.gl) * mc.vbas_time_series[i][j])) @ mc.WPP_time_series[i][j+1].T
						RHS = RHS @ np.diag(d_activation_list[j+1](mc.gbas / (mc.gbas + mc.gapi + mc.gl) * mc.vbas_time_series[i][j+1]))
						BPP = mc.BPP_time_series[i][j]
					

					cos = cos_sim(BPP, RHS)

					angle_BPP_RHS_arr.append(deg(cos))
					# print(angle_dWPP_arr[-1])

				mc.angle_BPP_RHS_time_series.append(angle_BPP_RHS_arr)

		return MC_list
