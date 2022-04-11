import numpy as np
from microcircuit import *

# define mappings from str input to functions
function_mappings = {
	'linear': linear,
    'relu': relu,
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

	uI_init = [np.random.normal(0, 1, size=layers[-1])]

	return uP_init, uI_init
		

# defines the weight inits
def init_weights(layers, seed):

	np.random.seed(seed)

	# forward pp weights: connects all layers k, k+1
	WPP_init = []
	for i in range(len(layers)-1):
	    WPP_init.append(np.random.uniform(0, 1, size=(layers[i+1], layers[i]))) # << init fw weights positively
	    
	# p to inter: connects N-1 to N
	WIP_init = [np.random.uniform(-1, 1, size=(layers[-1], layers[-2]))]

	# backwards p to p: connects N to k
	BPP_init = []
	for i in range(1, len(layers)-1):
	    BPP_init.append(np.random.uniform(-1, 0, size=(layers[i], layers[-1]))) # << init bw weights negatively
	    # BPP_init.append(np.linalg.pinv(WPP_init[i]))

	# backwards inter to p: connects N to k
	BPI_init = []
	for i in range(1, len(layers)-1):
	    BPI_init.append(np.random.uniform(-1, 1, size=(layers[i], layers[-1])))


	return WPP_init, WIP_init, BPP_init, BPI_init


# defines the microcircuit models
def init_MC(params, seeds):

	MC_list = []

	for seed in seeds:

		uP_init, uI_init = init_voltages(params["layers"], seed)
		WPP_init, WIP_init, BPP_init, BPI_init = init_weights(params["layers"], seed)

		# if a list of activations has been passed, use it
		if isinstance(params["activation"], list):
			activation_list = [function_mappings[activation] for activation in params["activation"]]
		# else, set same activation for all layers
		else:
			activation_list = function_mappings[params["activation"]]

		if params["model_type"] in ["DTPDRL", "LDRL"]:

			MC_list.append(
				phased_noise_model(
					dt=params["dt"],
					dtxi=params["dtxi"],
					tausyn=params["tausyn"],
					Tbw=params["Tbw"],
					Tpres=params["Tpres"],
					noise_scale=params["noise_scale"],
					alpha=params["alpha"],
					inter_low_pass=params["inter_low_pass"],
					pyr_hi_pass=params["pyr_hi_pass"],
					dWPP_low_pass=params["dWPP_low_pass"],
					gate_regularizer=params["gate_regularizer"],
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

					noise_deg=params["noise_deg"],
					taueps=params["taueps"]
					)
				)
			# save seed of mc and other params
			MC_list[-1].seed = seed
			MC_list[-1].dataset_size = params["dataset_size"]
			MC_list[-1].epochs = params["epochs"]
			MC_list[-1].init_in_SPS = params["init_in_SPS"]
			# data recording options
			MC_list[-1].rec_per_steps=params["rec_per_steps"]
			MC_list[-1].rec_MSE=params["rec_MSE"]
			MC_list[-1].rec_WPP=params["rec_WPP"]
			MC_list[-1].rec_WIP=params["rec_WIP"]
			MC_list[-1].rec_BPP=params["rec_BPP"] 
			MC_list[-1].rec_BPI=params["rec_BPI"]
			MC_list[-1].rec_uP=params["rec_uP"]
			MC_list[-1].rec_rP_breve=params["rec_rP_breve"] 
			MC_list[-1].rec_rP_breve_HI=params["rec_rP_breve_HI"]
			MC_list[-1].rec_uI=params["rec_uI"]
			MC_list[-1].rec_rI_breve=params["rec_rI_breve"]
			MC_list[-1].rec_vapi=params["rec_vapi"]
			MC_list[-1].rec_vapi_noise=params["rec_vapi_noise"]
			MC_list[-1].rec_noise=params["rec_noise"]
			MC_list[-1].rec_epsilon=params["rec_epsilon"]
			MC_list[-1].rec_epsilon_LO=params["rec_epsilon_LO"]
		

	for mc in MC_list:
		if mc.init_in_SPS:
			mc.set_self_predicting_state()

	return MC_list


