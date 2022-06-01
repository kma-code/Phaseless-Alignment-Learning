import numpy as np
from microcircuit import *
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


def compare_updates(MC_list, model):
	# compare updates of mc object to dWPP of BP

	# define the number of recorded time steps which belong to the pre-training
	# and therefore should be skipped in plotting
	TPRE = int(MC_list[0].settling_time / MC_list[0].dt / MC_list[0].rec_per_steps)

	if model == "BP":

		for mc in MC_list:
			d_activation_list = [derivative_mappings[activation] for activation in mc.activation]
			r0 = np.tile(mc.input, mc.epochs)

			mc.angle_dWPP_time_series = []

			# for every time step
			for i in range(len(mc.dWPP_time_series[TPRE:])):
				angle_dWPP_arr = []

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

					angle_dWPP_arr.append(deg(cos))
					# print(angle_dWPP_arr[-1])

				mc.angle_dWPP_time_series.append(angle_dWPP_arr)

		return MC_list


def compare_jacobians(MC_list, model):

	if model == "BP":

		for mc in MC_list:
			d_activation_list = [derivative_mappings[activation] for activation in mc.activation]

			mc.angle_jacobians_BP_time_series = []

			# for every time step
			for i in range(len(mc.dWPP_time_series)):
				angle_jacobians_BP_arr = []

				# for every hidden layer
				for j in range(len(mc.layers)-2):

					if MC_list[0].bw_connection_mode == 'skip':
						# construct Jacobian J_f^T
						J_f_T = np.eye(mc.layers[-1])
						# multiply phi' @ W.T from left
						for k in range(len(mc.layers)-2, j, -1):
							J_f_T = np.diag(d_activation_list[k-1](mc.gbas / (mc.gbas + mc.gl) * mc.vbas_time_series[i][k-1])) @ mc.WPP_time_series[i][k].T @ J_f_T

						# construct Jacobian J_g
						J_g = mc.BPP_time_series[i][j] @ np.diag(d_activation_list[-1](mc.gbas / (mc.gbas + mc.gl) * mc.vbas_time_series[i][-1]))

					elif MC_list[0].bw_connection_mode == 'layered':
						J_f_T = np.diag(d_activation_list[j](mc.gbas / (mc.gbas + mc.gl) * mc.vbas_time_series[i][j])) @ mc.WPP_time_series[i][j+1].T
						J_g   = mc.BPP_time_series[i][j] @ np.diag(d_activation_list[j+1](mc.gbas / (mc.gbas + mc.gl) * mc.vbas_time_series[i][j+1]))

					

					cos = cos_sim(J_g, J_f_T)

					angle_jacobians_BP_arr.append(deg(cos))
					# print(angle_dWPP_arr[-1])

				mc.angle_jacobians_BP_time_series.append(angle_jacobians_BP_arr)

		return MC_list