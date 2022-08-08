import numpy as np
from microcircuit import *

# define mappings from str input to functions
function_mappings = {
	'linear': linear,
    'relu': relu,
    'soft_relu': soft_relu,
    'logistic': logistic,
    'tanh': tanh,
   	'hard_sigmoid': hard_sigmoid
}


# defines the voltage inits
def init_voltages(layers, seed):

	np.random.seed(seed)

	uP_init = []
	for i in range(1, len(layers)):
	    uP_init.append(np.random.normal(0, 1, size=layers[i]))

	uI_init = []
	for i in range(1, len(layers)-1):
	    uI_init.append(np.random.normal(0, 1, size=layers[i+1]))

	return uP_init, uI_init
		

# defines the weight inits
def init_weights(layers, WPP_range, WIP_range, BPP_range, BPI_range, seed):

	np.random.seed(seed)

	# forward pp weights: connects all layers k, k+1
	WPP_init = []
	for i in range(len(layers)-1):
	    WPP_init.append(np.random.uniform(WPP_range[0], WPP_range[1], size=(layers[i+1], layers[i])))

	# p to inter: connects k to k
	WIP_init = []
	for i in range(1, len(layers)-1):
	    WIP_init.append(np.random.uniform(WIP_range[0], WIP_range[1], size=(layers[i+1], layers[i])))

	# backwards p to p: connects k+1 to k
	BPP_init = []
	for i in range(1, len(layers)-1):
	    BPP_init.append(np.random.uniform(BPP_range[0], BPP_range[1], size=(layers[i], layers[i+1])))

	# backwards inter to p: connects k to k
	BPI_init = []
	for i in range(1, len(layers)-1):
	    BPI_init.append(np.random.uniform(BPI_range[0], BPI_range[1], size=(layers[i], layers[i+1])))


	return WPP_init, WIP_init, BPP_init, BPI_init


# defines the microcircuit models
def init_MC(params, seeds, teacher=False):

	MC_list = []

	if teacher:
		# init a single teacher based on the first seed in seed list
		np.random.seed(seeds[0])
		seeds = [np.random.randint(100000, 999999)]

	for seed in seeds:

		if teacher:
			uP_init, uI_init = init_voltages(params["layers"], seed)
			WPP_init, WIP_init, BPP_init, BPI_init = init_weights(
				params["layers"],
				params["init_teacher_WPP_range"],
				params["init_teacher_WIP_range"],
				params["init_teacher_BPP_range"],
				params["init_teacher_BPI_range"],
				seed
				)

		else:
			uP_init, uI_init = init_voltages(params["layers"], seed)
			WPP_init, WIP_init, BPP_init, BPI_init = init_weights(
				params["layers"],
				params["init_WPP_range"],
				params["init_WIP_range"],
				params["init_BPP_range"],
				params["init_BPI_range"],
				seed
				)

		# if a list of activations has been passed, use it
		if isinstance(params["activation"], list):
			activation_list = [function_mappings[activation] for activation in params["activation"]]
		# else, set same activation for all layers
		else:
			activation_list = function_mappings[params["activation"]]

		if params["model_type"] in ["BP", "FA"]:
			MC_list.append(
				base_model(
					seed=seed,
					bw_connection_mode='layered',
					dWPP_use_activation=params["dWPP_use_activation"],
					dt=params["dt"],
					Tpres=params["Tpres"],
					model=params["model_type"],
		            activation=activation_list,
		            layers=params["layers"],

		            uP_init=uP_init,
		            uI_init=uI_init,

					WPP_init=WPP_init,
					WIP_init=WIP_init,
					BPP_init=BPP_init,
					BPI_init=BPI_init,

					gl=params["gl"],
					gden=params["gden"],
					gbas=params["gbas"],
					gapi=params["gapi"],
					gnI=params["gnI"],
					gntgt=params["gntgt"],
					eta_fw=params["eta_fw"],
					eta_bw=params["eta_bw"],
					eta_PI=params["eta_PI"],
					eta_IP=params["eta_IP"]
					)
				)

		elif params["model_type"] in ["DTPDRL", "LDRL"]:
			noise_deg = params["noise_deg"] if "noise_deg" in params else None
			taueps = params["taueps"] if "taueps" in params else None
			tauxi = params["tauxi"] if "tauxi" in params else None

			MC_list.append(
				noise_model(
					seed=seed,
					bw_connection_mode='layered',
					dWPP_use_activation=params["dWPP_use_activation"],
					dt=params["dt"],
					dtxi=params["dtxi"],
					tauHP=params["tauHP"],
					tauLO=params["tauLO"],
					Tpres=params["Tpres"],
					noise_scale=params["noise_scale"],
					alpha=params["alpha"],
					inter_low_pass=params["inter_low_pass"],
					pyr_hi_pass=params["pyr_hi_pass"],
					dWPP_low_pass=params["dWPP_low_pass"],
					dWPP_r_low_pass=params["dWPP_r_low_pass"],
					dWPP_post_low_pass=params["dWPP_post_low_pass"],
					gate_regularizer=params["gate_regularizer"],

					noise_type=params["noise_type"],
					noise_mode=params["noise_mode"],
					model=params["model_type"],
		            activation=activation_list,
		            layers=params["layers"],

		            uP_init=uP_init,
		            uI_init=uI_init,

					WPP_init=WPP_init,
					WIP_init=WIP_init,
					BPP_init=BPP_init,
					BPI_init=BPI_init,

					gl=params["gl"],
					gden=params["gden"],
					gbas=params["gbas"],
					gapi=params["gapi"],
					gnI=params["gnI"],
					gntgt=params["gntgt"],
					eta_fw=params["eta_fw"],
					eta_bw=params["eta_bw"],
					eta_PI=params["eta_PI"],
					eta_IP=params["eta_IP"],

					noise_deg=noise_deg,
					taueps=taueps,
					tauxi=tauxi

					)
				)
		# save seed of mc and other params
		MC_list[-1].seed = seed
		MC_list[-1].input_signal = params["input_signal"]
		MC_list[-1].dataset_size = params["dataset_size"]
		MC_list[-1].settling_time = params["settling_time"]
		MC_list[-1].copy_teacher_weights = params["copy_teacher_weights"]
		MC_list[-1].copy_teacher_voltages = params["copy_teacher_voltages"]
		if teacher:
			MC_list[-1].epochs = 1
		else:
			MC_list[-1].epochs = params["epochs"]
		MC_list[-1].init_in_SPS = params["init_in_SPS"]
		# data recording options
		MC_list[-1].rec_per_steps=params["rec_per_steps"]
		if teacher:
			MC_list[-1].rec_MSE=False
			MC_list[-1].rec_error=False
		else:
			MC_list[-1].rec_MSE=params["rec_MSE"]
			MC_list[-1].rec_error=params["rec_error"]
		MC_list[-1].rec_input=params["rec_input"]
		MC_list[-1].rec_target=params["rec_target"]
		MC_list[-1].rec_WPP=params["rec_WPP"]
		MC_list[-1].rec_WIP=params["rec_WIP"]
		MC_list[-1].rec_BPP=params["rec_BPP"] 
		MC_list[-1].rec_BPI=params["rec_BPI"]
		MC_list[-1].rec_dWPP=params["rec_dWPP"]
		MC_list[-1].rec_dWIP=params["rec_dWIP"]
		MC_list[-1].rec_dBPP=params["rec_dBPP"] 
		MC_list[-1].rec_dBPI=params["rec_dBPI"]
		MC_list[-1].rec_uP=params["rec_uP"]
		MC_list[-1].rec_uP_breve=params["rec_uP_breve"]
		MC_list[-1].rec_rP_breve=params["rec_rP_breve"]
		MC_list[-1].rec_uI=params["rec_uI"]
		MC_list[-1].rec_uI_breve=params["rec_uI_breve"]
		MC_list[-1].rec_rI_breve=params["rec_rI_breve"]
		MC_list[-1].rec_vapi=params["rec_vapi"]
		MC_list[-1].rec_vbas=params["rec_vbas"]
		# some variables only exist in DTPDRL
		if params["model_type"] in ["LDRL", "DTPDRL"]:
			MC_list[-1].rec_rP_breve_HI=params["rec_rP_breve_HI"]
			MC_list[-1].rec_vapi_noise=params["rec_vapi_noise"]
			MC_list[-1].rec_noise=params["rec_noise"]
			MC_list[-1].rec_epsilon=params["rec_epsilon"]
			MC_list[-1].rec_epsilon_LO=params["rec_epsilon_LO"]
		else:
			MC_list[-1].rec_rP_breve_HI=False
			MC_list[-1].rec_vapi_noise=False
			MC_list[-1].rec_noise=False
			MC_list[-1].rec_epsilon=False
			MC_list[-1].rec_epsilon_LO=False
		if teacher:
			MC_list[-1].rec_uP_breve=True
		

	for mc in MC_list:
		if mc.init_in_SPS:
			mc.set_self_predicting_state()

	return MC_list


