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

	if model == "BP":

		for mc in MC_list:
			d_activation_list = [derivative_mappings[activation] for activation in mc.activation]

			mc.angle_dWPP_time_series = []

			# for every time step
			for i in range(len(mc.dWPP_time_series)):
				angle_dWPP_arr = []

				# for every layer
				for j in range(len(mc.layers)-1):
					if j == 0:
						r_in = mc.input[0]
					else:
						r_in = mc.rP_breve_time_series[i][j-1]

					# construct weight update for BP net

					# for output layer
					dWPP_BP = np.outer(mc.error_time_series[i], r_in)

					# multiply phi' @ W.T from left
					for k in range(len(mc.layers)-2, j, -1):
						dWPP_BP = np.diag(d_activation_list[k-1](mc.uP_breve_time_series[i][k-1])) @ mc.WPP_time_series[i][k].T @ dWPP_BP

					cos = cos_sim(mc.dWPP_time_series[i][j], dWPP_BP)

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
					
					# construct Jacobian J_f^T
					J_f_T = np.eye(mc.layers[-1])
					# multiply phi' @ W.T from left
					for k in range(len(mc.layers)-2, j, -1):
						J_f_T = np.diag(d_activation_list[k-1](mc.uP_breve_time_series[i][k-1])) @ mc.WPP_time_series[i][k].T @ J_f_T

					# construct Jacobian J_g
					J_g = mc.BPP_time_series[i][j] @ np.diag(d_activation_list[-1](mc.uP_breve_time_series[i][-1]))

					cos = cos_sim(J_g, J_f_T)

					angle_jacobians_BP_arr.append(deg(cos))
					# print(angle_dWPP_arr[-1])

				mc.angle_jacobians_BP_time_series.append(angle_jacobians_BP_arr)

		return MC_list
