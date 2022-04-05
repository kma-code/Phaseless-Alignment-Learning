import numpy as np
import copy
import inspect
from scipy import signal
#import matplotlib.pyplot as plt

# define activation functions
def linear(x):
	return np.array(x)
def d_linear(x):
    return np.ones_like(x)

def relu(x):
    return np.maximum(x, 0, np.array(x))
def d_relu(x):
    return np.heaviside(x, 0)

def logistic(x):
    return 1/(1 + np.exp(-x))
def d_logistic(x):
    y = logistic(x)
    return y * (1.0 - y)

def tanh(x):
    return np.tanh(x)
def d_tanh(x):
    y = tanh(x)
    return 1 - y**2

def hard_sigmoid(x):
	# see defintion at torch.nn.Hardsigmoid
	return np.maximum(0, np.minimum(1, x/6 + 1/2), np.array(x))

def d_hard_sigmoid(x):
	return np.heaviside(x/6 + 1/2, 0) * np.heaviside(1 - (x/6 + 1/2), 0)

# cosine similarity between tensors
def cos_sim(A, B):
	if A.ndim == 1 and B.ndim == 1:
		return A.T @ B / np.linalg.norm(A) / np.linalg.norm(B)
	else:
		return np.trace(A.T @ B) / np.linalg.norm(A) / np.linalg.norm(B)

def dist(A, B):
	return np.linalg.norm(A-B)

def deg(cos):
	# calculates angle in deg from cosine
	return np.arccos(cos) * 180 / np.pi
		


def deepcopy_array(array):
	""" makes a deep copy of an array of np-arrays """
	out = [nparray.copy() for nparray in array]
	return out.copy()

def MSE(output, target):
	return np.linalg.norm(output - target)**2

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


class base_model:
	""" This class implements a generic microcircuit model """

	def __init__(self, dt, Tpres, model, activation, layers,
					uP_init, uI_init, WPP_init, WIP_init, BPP_init, BPI_init,
					gl, gden, gbas, gapi, gnI, gntgt,
					eta_fw, eta_bw, eta_PI, eta_IP):

		self.model = model # FA, BP or PBP
		self.layers = layers
		self.uP = deepcopy_array(uP_init)
		self.uI = deepcopy_array(uI_init)
		# we also set up a buffer of voltages,
		# which corresponds to the value at the last time step
		self.uP_old = deepcopy_array(self.uP)
		self.uI_old = deepcopy_array(self.uI)

		# if a list of activations has been passed, use it
		if isinstance(activation, list):
			self.activation = activation
		# else, set same activation for all layers
		else:
			self.activation = [activation for layer in layers[1:]]


		# define the compartment voltages
		self.vbas = [np.zeros_like(uP) for uP in self.uP]
		self.vden = [np.zeros_like(uI) for uI in self.uI]
		self.vapi = [np.zeros_like(uP) for uP in self.uP[:-1]]
		# and make copies
		self.vbas_old = [np.zeros_like(uP) for uP in self.uP]
		self.vden_old = [np.zeros_like(uI) for uI in self.uI]
		self.vapi_old = [np.zeros_like(uP) for uP in self.uP[:-1]]

		self.WPP = deepcopy_array(WPP_init)
		self.WIP = deepcopy_array(WIP_init)
		self.BPP = deepcopy_array(BPP_init)
		self.BPI = deepcopy_array(BPI_init)

		if self.model == "BP":
			self.set_weights(BPP = [WPP.T for WPP in self.WPP[1:]])

		self.gl = gl
		self.gden = gden
		self.gbas = gbas
		self.gapi = gapi
		self.gnI = gnI
		self.gntgt = gntgt

		self.Time = 0 # initialize a model timer
		self.dt = dt
		self.Tpres = Tpres
		self.taueffP, self.taueffP_notgt, self.taueffI = self.calc_taueff()

		# learning rates
		self.eta_fw = eta_fw
		self.eta_bw = eta_bw
		self.eta_IP = eta_IP
		self.eta_PI = eta_PI

		# calculate lookahead
		self.uP_breve = [self.prospective_voltage(self.uP[i], self.uP_old[i], self.taueffP[i]) for i in range(len(self.uP))]
		self.uI_breve = [self.prospective_voltage(self.uI[i], self.uI_old[i], self.taueffI[i]) for i in range(len(self.uI))]
		# calculate rate of lookahead: phi(ubreve)
		self.rP_breve = [self.activation[i](self.uP_breve[i]) for i in range(len(self.uP_breve))]
		self.rI_breve = [self.activation[-1](self.uI_breve[-1])]

		self.r0 = np.zeros(self.layers[0])

	def init_record(self, rec_per_steps=1, rec_MSE=False, rec_WPP=False, rec_WIP=False, rec_BPP=False, rec_BPI=False,
		rec_uP=False, rec_rP_breve=False, rec_rP_breve_HI=False, rec_uI=False, rec_rI_breve=False, rec_vapi=False, rec_vapi_noise=False, rec_noise=False, rec_epsilon=False, rec_epsilon_LO=False):
		# records the values of the variables given in var_array
		# e.g. WPP, BPP, uP_breve
		# rec_per_steps sets after how many steps data is recorded
		
		
		if rec_MSE:
			self.MSE_time_series = []
		if rec_WPP:
			self.WPP_time_series = []
		if rec_WIP:
			self.WIP_time_series = []
		if rec_BPP:
			self.BPP_time_series = []
		if rec_BPI:
			self.BPI_time_series = []
		if rec_uP:
			self.uP_time_series = []
		if rec_rP_breve:
			self.rP_breve_time_series = []
		if rec_rP_breve_HI:
			self.rP_breve_HI_time_series = []
		if rec_uI:
			self.uI_time_series = []
		if rec_rI_breve:
			self.rI_breve_time_series = []
		if rec_vapi:
			self.vapi_time_series = []
		if rec_vapi_noise:
			self.vapi_noise_time_series = []
		if rec_noise:
			self.noise_time_series = []
		if rec_epsilon:
			self.epsilon_time_series = []
		if rec_epsilon_LO:
			self.epsilon_LO_time_series = []

		self.rec_per_steps = rec_per_steps
		self.rec_counter = 0


	def record_step(self, target=None):
		if hasattr(self, 'MSE_time_series') and target is not None:
			self.MSE_time_series.append(
				MSE(self.uP_breve[-1], target)
				)
		if hasattr(self, 'WPP_time_series'):
			self.WPP_time_series.append(copy.deepcopy(self.WPP))
		if hasattr(self, 'WIP_time_series'):
			self.WIP_time_series.append(copy.deepcopy(self.WIP))
		if hasattr(self, 'BPP_time_series'):
			self.BPP_time_series.append(copy.deepcopy(self.BPP))
		if hasattr(self, 'BPI_time_series'):
			self.BPI_time_series.append(copy.deepcopy(self.BPI))
		if hasattr(self, 'uP_time_series'):
			self.uP_time_series.append(copy.deepcopy(self.uP))
		if hasattr(self, 'rP_breve_time_series'):
			self.rP_breve_time_series.append(copy.deepcopy(self.rP_breve))
		if hasattr(self, 'rP_breve_HI_time_series'):
			self.rP_breve_HI_time_series.append(copy.deepcopy(self.rP_breve_HI))
		if hasattr(self, 'uI_time_series'):
			self.uI_time_series.append(copy.deepcopy(self.uI))
		if hasattr(self, 'rI_breve_time_series'):
			self.rI_breve_time_series.append(copy.deepcopy(self.rI_breve))
		if hasattr(self, 'vapi_time_series'):
			self.vapi_time_series.append(copy.deepcopy(self.vapi))
		if hasattr(self, 'vapi_noise_time_series'):
			self.vapi_noise_time_series.append(copy.deepcopy(self.vapi_noise))
		if hasattr(self, 'noise_time_series'):
			self.noise_time_series.append(copy.deepcopy(self.noise))
		if hasattr(self, 'epsilon_time_series'):
			self.epsilon_time_series.append(copy.deepcopy(self.epsilon))
		if hasattr(self, 'epsilon_LO_time_series'):
			self.epsilon_LO_time_series.append(copy.deepcopy(self.epsilon_LO))


	def calc_taueff(self):
		# calculate tau_eff for pyramidals and interneuron
		# taueffP is one value per layer

		taueffP = []
		for i in self.uP:
			taueffP.append(1 / (self.gl + self.gbas + self.gapi))
		taueffP[-1] = 1 / (self.gl + self.gbas + self.gntgt)
		# tau_eff for output layer in absence of target
		taueffP_notgt = [1 / (self.gl + self.gbas)]

		taueffI = [1 / (self.gl + self.gden + self.gnI)]

		return taueffP, taueffP_notgt, taueffI


	def get_conductances(self):
		return self.gl, self.gden, self.gbas, self.gapi, self.gnI, self.gntgt


	def get_weights(self):
		return self.WPP, self.WIP, self.BPP, self.BPI


	def set_weights(self, model=None, WPP=None, WIP=None, BPP=None, BPI=None):
		# if another model is given, copy its weights
		if hasattr(model, '__dict__'):
			WPP, WIP, BPP, BPI = model.get_weights()
			print(f"Copying weights from model {model}")

		if WPP is not None: self.WPP = deepcopy_array(WPP)
		if WIP is not None: self.WIP = deepcopy_array(WIP)
		if BPP is not None: self.BPP = deepcopy_array(BPP)
		if BPI is not None: self.BPI = deepcopy_array(BPI)


	def set_self_predicting_state(self):

		# set WIP and BPI to values corresponding to self-predicting state
		for i in range(0, len(self.BPI)):
			self.BPI[i] = - self.BPP[i].copy()
		self.WIP[-1] = self.gbas * (self.gl + self.gden) / (self.gden * (self.gl + self.gbas)) * self.WPP[-1].copy()

	def get_voltages(self):
		return self.uP, self.uI

	def get_old_voltages(self):
		return self.uP_old, self.uI_old


	def set_voltages(self, model=None, uP=None, uP_old=None, uI=None, uI_old=None):
		# if another model is given, copy its voltages
		if hasattr(model, '__dict__'):
			uP, uI = model.get_voltages()
			uP_old, uI_old = model.get_old_voltages()
			print(f"Copying voltages from model {model}")

		if uP is not None:
			for i in range(len(self.layers)-1):
				self.uP[i] = copy.deepcopy(uP[i])

		if uP_old is not None:
			for i in range(len(self.layers)-1):
				self.uP_old[i] = copy.deepcopy(uP_old[i])

		if uI is not None:
			self.uI = copy.deepcopy(uI)
		if uI_old is not None:
			self.uI_old = copy.deepcopy(uI_old)

	def calc_vapi(self, rPvec, BPP_mat, rIvec, BPI_mat):
		# returns apical voltages in pyramidals of a given layer
		# input: rPvec: vector of rates from pyramidal voltages in output layer
		# 		 WPP_mat: matrix connecting pyramidal to pyramidal
		# 		 rIvec: vector of rates from interneuron voltages in output layer
		# 		 BPI_mat: matrix connecting interneurons to pyramidals

		return BPP_mat @ rPvec + BPI_mat @ rIvec

	def calc_vbas(self, rPvec, WPP_mat):
		# returns basal voltages in pyramidals of a given layer
		# input: rPvec: vector of rates from pyramidal voltages in layer below
		# 		 WPP_mat: matrix connecting pyramidal to pyramidal

		return WPP_mat @ rPvec

	def calc_vden(self, rPvec, WIP_mat):
		# returns dendritic voltages in inteneurons
		# input: rPvec: vector of rates from pyramidal voltages in layer below
		# 		 WIP_mat: matrix connecting pyramidal to pyramidal

		return WIP_mat @ rPvec


	def prospective_voltage(self, uvec, uvec_old, tau, dt=None):
		# returns an approximation of the lookahead of voltage vector u at current time
		if dt == None:
			dt = self.dt
		return uvec_old + tau * (uvec - uvec_old) / dt


	def evolve_system(self, r0=None, u_tgt=None, learn_weights=True):

		""" evolves the system by one time step:
			updates synaptic weights and voltages
			given input rate r0
		"""

		# increase timer by dt and round float to nearest dt
		self.Time = np.round(self.Time + self.dt, decimals=np.int(np.round(-np.log10(self.dt))))

		self.duP, self.duI = self.evolve_voltages(r0, u_tgt) # includes recalc of rP_breve
		if learn_weights:
			self.dWPP, self.dWIP, self.dBPP, self.dBPI = self.evolve_synapses(r0)

		# apply evolution
		for i in range(len(self.duP)):
			self.uP[i] += self.duP[i]
		for i in range(len(self.duI)):
			self.uI [i]+= self.duI[i]

		if learn_weights:
			for i in range(len(self.dWPP)):
				self.WPP[i] += self.dWPP[i]
			for i in range(len(self.dWIP)):
				self.WIP[i] += self.dWIP[i]
			for i in range(len(self.dBPP)):
				self.BPP[i] += self.dBPP[i]
			for i in range(len(self.dBPI)):
				self.BPI[i] += self.dBPI[i]

		# record step
		if hasattr(self, 'rec_per_steps'):
			self.rec_counter += 1
			if self.rec_counter % self.rec_per_steps == 0:
				self.rec_counter = 0
				self.record_step(target=u_tgt)


	def evolve_voltages(self, r0=None, u_tgt=None):
		""" Evolves the pyramidal and interneuron voltages by one dt """
		""" using r0 as input rates """

		self.duP = [np.zeros(shape=uP.shape) for uP in self.uP]
		self.duI = [np.zeros(shape=uI.shape) for uI in self.uI]

		# same for dendritic voltages and rates
		self.rP_breve_old = deepcopy_array(self.rP_breve)
		self.rI_breve_old = deepcopy_array(self.rI_breve)
		if self.r0 is not None:
			self.r0_old = self.r0.copy()
		self.vbas_old = deepcopy_array(self.vbas)
		self.vden_old = deepcopy_array(self.vden)
		self.vapi_old = deepcopy_array(self.vapi)

		# calculate lookahead
		self.uP_breve = [self.prospective_voltage(self.uP[i], self.uP_old[i], self.taueffP[i]) for i in range(len(self.uP))]
		self.uI_breve = [self.prospective_voltage(self.uI[i], self.uI_old[i], self.taueffI[i]) for i in range(len(self.uI))]
		# calculate rate of lookahead: phi(ubreve)
		self.rP_breve = [self.activation[i](self.uP_breve[i]) for i in range(len(self.uP_breve))]
		self.rI_breve = [self.activation[-1](self.uI_breve[-1])]
		self.r0 = r0

		# before modifying uP and uI, we need to save copies
		# for future calculation of u_breve
		self.uP_old = deepcopy_array(self.uP)
		self.uI_old = deepcopy_array(self.uI)

		self.vbas, self.vapi, self.vden = self.calc_dendritic_updates(r0, u_tgt)
		self.duP, self.duI = self.calc_somatic_updates(u_tgt)

		return self.duP, self.duI

	
	def calc_dendritic_updates(self, r0=None, u_tgt=None):

		# calculate dendritic voltages from lookahead
		if r0 is not None:
			self.vbas[0] = self.WPP[0] @ self.r0

		for i in range(1, len(self.layers)-1):
			self.vbas[i] = self.calc_vbas(self.rP_breve[i-1], self.WPP[i])

		self.vden[0] = self.calc_vden(self.rP_breve[-2], self.WIP[-1])

		for i in range(0, len(self.layers)-2):
			self.vapi[i] = self.calc_vapi(self.rP_breve[-1], self.BPP[i], self.rI_breve[-1], self.BPI[i])

		return self.vbas, self.vapi, self.vden




	def calc_somatic_updates(self, u_tgt=None):
		"""
			calculates somatic updates from dendritic potentials

		"""

		# update somatic potentials
		ueffI = self.taueffI[-1] * (self.gden * self.vden[-1] + self.gnI * self.uP_breve[-1])
		delta_uI = (ueffI - self.uI[-1]) / self.taueffI[-1]
		self.duI[-1] = self.dt * delta_uI

		for i in range(0, len(self.layers)-2):
			ueffP = self.taueffP[i] * (self.gbas * self.vbas[i] + self.gapi * self.vapi[i])
			delta_uP = (ueffP - self.uP[i]) / self.taueffP[i]
			self.duP[i] = self.dt * delta_uP

		if u_tgt is not None:
			ueffP = self.taueffP[-1] * (self.gbas * self.vbas[-1] + self.gntgt * u_tgt[-1])
			delta_uP = (ueffP - self.uP[-1]) / self.taueffP[-1]
		else:
			ueffP = self.taueffP_notgt[-1] * (self.gbas * self.vbas[-1])
			delta_uP = (ueffP - self.uP[-1]) / self.taueffP[-1]
		self.duP[-1] = self.dt * delta_uP

		return self.duP, self.duI

		


	def evolve_synapses(self, r0):
		
		""" evolves all synapses by a dt """


		"""
			plasticity of WPP

		"""

		self.dWPP = [np.zeros(shape=WPP.shape) for WPP in self.WPP]
		self.dWIP = [np.zeros(shape=WIP.shape) for WIP in self.WIP]
		self.dBPP = [np.zeros(shape=BPP.shape) for BPP in self.BPP]
		self.dBPI = [np.zeros(shape=BPI.shape) for BPI in self.BPI]

		# input layer
		if r0 is not None:
			# print("updating WPP0")
			self.dWPP[0] = self.dt * self.eta_fw[0] * np.outer(
					self.rP_breve[0] - self.activation[0](self.gbas / (self.gl + self.gbas + self.gapi) * self.vbas_old[0]),
													self.r0_old)
		# hidden layers
		for i in range(1, len(self.WPP)-1):
			# print(f"updating WPP{i}")
			self.dWPP[i] = self.dt * self.eta_fw[i] * np.outer(
					self.rP_breve[i] - self.activation[i](self.gbas / (self.gl + self.gbas + self.gapi) * self.vbas_old[i]),
													self.rP_breve_old[i-1])
		# output layer
		# print("updating WPP-1")
		self.dWPP[-1] = self.dt * self.eta_fw[-1] * np.outer(
					self.rP_breve[-1] - self.activation[-1](self.gbas / (self.gl + self.gbas) * self.vbas_old[-1]),
													self.rP_breve_old[-2])

		"""
			plasticity of WIP

		"""

		self.dWIP[-1] = self.dt * self.eta_IP[-1] * np.outer(
					self.rI_breve[-1] - self.activation[-1](self.gden / (self.gl + self.gden) * self.vden_old[-1]),
													self.rP_breve_old[-2])

		"""
			plasticity of BPI

		"""

		for i in range(0, len(self.BPI)):
			if self.eta_PI[i] != 0:
				self.dBPI[i] = self.dt * self.eta_PI[i] * np.outer(-self.vapi[i], self.rI_breve_old[-1])

		"""
			plasticity of BPP

		"""

		if self.model == 'FA':
			# do nothing
			0

		elif self.model == 'BP':
			self.set_weights(BPP = [WPP.T for WPP in self.WPP[1:]])


		return self.dWPP, self.dWIP, self.dBPP, self.dBPI 




class phased_noise_model(base_model):
	""" This class inherits all properties from the base model class and adds the function to add phased noise """
	def __init__(self, dt, dtxi, tausyn, Tbw, Tpres, noise_scale, alpha, inter_low_pass, pyr_hi_pass, dWPP_low_pass, gate_regularizer, noise_mode,
					model, activation, layers,
					uP_init, uI_init, WPP_init, WIP_init, BPP_init, BPI_init,
					gl, gden, gbas, gapi, gnI, gntgt,
					eta_fw, eta_bw, eta_PI, eta_IP):

		# init base_model with same settings
		super().__init__(dt=dt, Tpres=Tpres,
			model=model, activation=activation, layers=layers,
            uP_init=uP_init, uI_init=uI_init,
            WPP_init=WPP_init, WIP_init=WIP_init, BPP_init=BPP_init, BPI_init=BPI_init,
            gl=gl, gden=gden, gbas=gbas, gapi=gapi, gnI=gnI, gntgt=gntgt,
            eta_fw=eta_fw, eta_bw=eta_bw, eta_PI=eta_PI, eta_IP=eta_IP)

		# new variables:

		# mode of noise injection (order vapi or uP or uP_adative)
		self.noise_mode = noise_mode
		# for uP_adaptive, we need epsilon: measures angle between BPP, WPP.T
		self.epsilon = [np.float64(1.0) for BPP in self.BPP]
		if noise_mode == 'dynamic':
			self.depsilon = [np.float64(0.0) for BPP in self.BPP]
			self.dBPP = [np.zeros(shape=BPP.shape) for BPP in self.BPP]
		# low-pass filtered version of epsilon
		self.epsilon_LO = deepcopy_array(self.epsilon)

		# whether to low-pass filter the interneuron dendritic input
		self.inter_low_pass = inter_low_pass
		# whether to high-pass filter rPbreve for updates of BPP
		self.pyr_hi_pass = pyr_hi_pass
		# whether to low-pass filter updates of WPP
		self.dWPP_low_pass = dWPP_low_pass
		# whether to gate application of the regularizer
		self.gate_regularizer = gate_regularizer
		if self.gate_regularizer:
			# need to to define the gate, i.e. the derivative of activation
			self.d_activation = []
			for activation in self.activation:
				if activation == relu:
					self.d_activation.append(d_relu)
				elif activation == hard_sigmoid:
					self.d_activation.append(d_hard_sigmoid)
		# noise time scale
		self.dtxi = dtxi
		# decimals of dt
		self.dt_decimals = np.int(np.round(-np.log10(self.dt)))
		# synaptic time constant (sets the low-pass filter of interneuron)
		self.tausyn = tausyn
		# time scale of backward learning phase
		self.Tbw = Tbw
		# define a variable for currently learnt backwards synapse
		# init at last synapse
		self.active_bw_syn = 0
		# gaussian noise properties
		self.noise_scale = noise_scale
		self.noise = [np.zeros(shape=uP.shape) for uP in self.uP]
		# self.noise_breve = [np.zeros(shape=uP.shape) for uP in self.uP]

		# we need a new variable: vapi after noise has been added
		# i.e. vapi = BPP rP + BPI rI (as usual), and vapi_noise = vapi + noise
		self.vapi_noise = deepcopy_array(self.vapi)

		# init a counter for time steps after which to resample noise
		self.noise_counter = 0
		self.noise_total_counts = np.round(self.dtxi / self.dt, decimals=self.dt_decimals)

		# init a high-pass filtered version of rP_breve
		self.rP_breve_HI = deepcopy_array(self.rP_breve)

		# init a low-pass filtered version of dWPP
		self.dWPP_LO = [np.zeros(shape=WPP.shape) for WPP in self.WPP]

		# regularizer for backward weights
		self.alpha = alpha


	def evolve_system(self, r0=None, u_tgt=None, learn_weights=True, learn_bw_weights=True):

		""" 
			This overwrites the vanilla system evolution and implements
			phased noise
		"""

		# update which backwards weights to learn
		if learn_bw_weights and self.Time % self.Tbw == 0:
			# print(f"Current time: {self.Time}s")
			self.active_bw_syn = 0 if self.active_bw_syn == len(self.BPP) - 1 else self.active_bw_syn + 1
			# print(f"Learning backward weights to layer {self.active_bw_syn + 1}")
			self.noise_counter = 0

		# calculate voltage evolution, including low pass on interneuron synapses
		# see calc_dendritic updates below
		self.duP, self.duI = self.evolve_voltages(r0, u_tgt, inject_noise=learn_bw_weights) # includes recalc of rP_breve

		# calculate hi-passed rP_breve for synapse BPP
		self.rP_breve_HI = self.calc_rP_breve_HI()

		if learn_weights:
			self.dWPP, self.dWIP, _, self.dBPI = self.evolve_synapses(r0)
		if learn_bw_weights:
			self.dBPP = self.evolve_bw_synapses(self.active_bw_syn)

		# apply evolution
		for i in range(len(self.duP)):
			self.uP[i] += self.duP[i]
		for i in range(len(self.duI)):
			self.uI [i]+= self.duI[i]

		if learn_weights:
			if self.dWPP_low_pass:
				# calculate lo-passed update of WPP
				self.dWPP_LO = self.calc_dWPP_LO()
				for i in range(len(self.dWPP_LO)):
					self.WPP[i] += self.dWPP_LO[i]
			else:
				for i in range(len(self.dWPP)):
					self.WPP[i] += self.dWPP[i]
			for i in range(len(self.dWIP)):
				self.WIP[i] += self.dWIP[i]
			for i in range(len(self.dBPI)):
				self.BPI[i] += self.dBPI[i]
		if learn_bw_weights:
			for i in range(len(self.dBPP)):
				self.BPP[i] += self.dBPP[i]

		# record step
		if hasattr(self, 'rec_per_steps'):
			self.rec_counter += 1
			if self.rec_counter % self.rec_per_steps == 0:
				self.rec_counter = 0
				self.record_step(target=u_tgt)

		# increase timer
		self.Time = np.round(self.Time + self.dt, decimals=self.dt_decimals)


	def evolve_voltages(self, r0=None, u_tgt=None, inject_noise=False):
		""" 
			Overwrites voltage evolution:
			Evolves the pyramidal and interneuron voltages by one dt
			using r0 as input rates
			>> Injects noise into vapi or uP
		"""

		self.duP = [np.zeros(shape=uP.shape) for uP in self.uP]
		self.duI = [np.zeros(shape=uI.shape) for uI in self.uI]

		# same for dendritic voltages and rates
		self.rP_breve_old = deepcopy_array(self.rP_breve)
		self.rI_breve_old = deepcopy_array(self.rI_breve)
		if self.r0 is not None:
			self.r0_old = self.r0.copy()
		self.vbas_old = deepcopy_array(self.vbas)
		self.vden_old = deepcopy_array(self.vden)
		self.vapi_old = deepcopy_array(self.vapi)

		# calculate lookahead
		self.uP_breve = [self.prospective_voltage(self.uP[i], self.uP_old[i], self.taueffP[i]) for i in range(len(self.uP))]
		self.uI_breve = [self.prospective_voltage(self.uI[i], self.uI_old[i], self.taueffI[i]) for i in range(len(self.uI))]
		# calculate rate of lookahead: phi(ubreve)
		self.rP_breve = [self.activation[i](self.uP_breve[i]) for i in range(len(self.uP_breve))]
		self.rI_breve = [self.activation[-1](self.uI_breve[-1])]
		self.r0 = r0

		# before modifying uP and uI, we need to save copies
		# for future calculation of u_breve
		self.uP_old = deepcopy_array(self.uP)
		self.uI_old = deepcopy_array(self.uI)

		self.vbas, self.vapi, self.vden = self.calc_dendritic_updates(r0, u_tgt)

		# inject noise into newly calculated vapi before calculating update du
		if inject_noise: self.inject_noise(layer=self.active_bw_syn, noise_scale=self.noise_scale)

		self.duP, self.duI = self.calc_somatic_updates(u_tgt, inject_noise=inject_noise)

		return self.duP, self.duI


	def calc_somatic_updates(self, u_tgt=None, inject_noise=False):
		"""
			this overwrites the somatic update rules
			only difference to super: uses vapi_noise instead of vapi

		"""

		# update somatic potentials
		ueffI = self.taueffI[-1] * (self.gden * self.vden[-1] + self.gnI * self.uP_breve[-1])
		delta_uI = (ueffI - self.uI[-1]) / self.taueffI[-1]
		self.duI[-1] = self.dt * delta_uI

		for i in range(0, len(self.layers)-2):
			if inject_noise:
				ueffP = self.taueffP[i] * (self.gbas * self.vbas[i] + self.gapi * self.vapi_noise[i])
			else:
				ueffP = self.taueffP[i] * (self.gbas * self.vbas[i] + self.gapi * self.vapi[i])
			delta_uP = (ueffP - self.uP[i]) / self.taueffP[i]
			self.duP[i] = self.dt * delta_uP

		if u_tgt is not None:
			ueffP = self.taueffP[-1] * (self.gbas * self.vbas[-1] + self.gntgt * u_tgt[-1])
			delta_uP = (ueffP - self.uP[-1]) / self.taueffP[-1]
		else:
			ueffP = self.taueffP_notgt[-1] * (self.gbas * self.vbas[-1])
			delta_uP = (ueffP - self.uP[-1]) / self.taueffP[-1]
		self.duP[-1] = self.dt * delta_uP

		return self.duP, self.duI


	def inject_noise(self, layer, noise_scale):

		"""
			 this function injects noise into a given layer
			 by adding it to the apical potential
		"""
		
		# TO DO: adapt epsilon for multiple hidden layers

		# if dtxi timesteps have passed, sample new noise
		if np.all(self.noise[layer] == 0) or self.noise_counter % self.noise_total_counts == 0:

			if self.noise_mode == 'uP_adaptive':
				# if noise is non-zero:
				if np.linalg.norm(self.noise[layer]) != 0.0:# and self.epsilon[0] > 1e-5:

					# print("calculating epsilon, time:", self.Time)
					# calculate Jacobian alignment factor epsilon
					self.epsilon = [1/2 * (1 - self.noise[layer] @ self.BPP[layer] @ self.rP_breve_HI[-1]  \
					/ np.linalg.norm(self.noise[layer]) / np.linalg.norm(self.BPP[layer] @ self.rP_breve_HI[-1]))]

				# update low-pass filtered version of epsilon
				# self.epsilon_LO[layer] += self.dt / self.tausyn * (self.epsilon[layer] - self.epsilon_LO[layer])
				self.epsilon_LO[layer] += 1/1000 * (self.epsilon[layer] - self.epsilon_LO[layer])

				# generate noise, rescaled with epsilon
				self.noise[layer] = noise_scale[layer] * self.epsilon_LO[layer] * np.array([np.random.normal(0, np.abs(x)) for x in self.uP[layer]])

				# if epsilon is below threshold, do not inject noise
				if self.epsilon_LO[layer] > 1/2 * (1 - np.cos(20 * np.pi/180)): # use 20 deg as threshold
					self.vapi_noise[layer] = self.vapi[layer] + self.noise[layer]
				else:
					self.vapi_noise[layer] = self.vapi[layer]

			elif self.noise_mode == 'dynamic':

				# epsilon follows a diff eq in this case
				self.depsilon[layer] = self.dt/(1000 * self.dt) * (0.01 * self.vapi[layer] - self.epsilon[layer] + min([0.1, 1e12 * np.linalg.norm(self.dBPP[layer])/self.dt]) * self.uP[layer])
				self.epsilon[layer] += self.depsilon[layer]

				# generate noise, where epsilon sets the scale
				self.noise[layer] = np.array([np.random.normal(0, np.abs(x)) for x in self.epsilon[layer]])


			elif self.noise_mode == 'uP':
				# add noise with magnitude of rescaled uP
				self.noise[layer] = noise_scale[layer] * np.array([np.random.normal(0, np.abs(x)) for x in self.uP[layer]])
				self.vapi_noise[layer] = self.vapi[layer] + self.noise[layer]

			elif self.noise_mode == 'vapi':
				# add noise with magnitude of rescaled vapi
				self.noise[layer] = noise_scale[layer] * np.array([np.random.normal(0, np.abs(x)) for x in self.vapi[layer]])
				self.vapi_noise[layer] = self.vapi[layer] + self.noise[layer]
			
			self.noise_counter = 0

		self.noise_counter += 1

	def calc_rP_breve_HI(self):
		# updates the high-passed instantaneous rate rP_breve_HI which is used to update BPP
		# High-pass has the form d v_out = d v_in - dt/tau * v_out

		# we will only need rP_breve_HI coming from the final layer, so we freeze the others
		# for i in range(len(self.rP_breve)):
		#       self.rP_breve_HI[i] += (self.rP_breve[i] - self.rP_breve_old[i]) - self.dt / self.tausyn * self.rP_breve_HI[i]
		self.rP_breve_HI[-1] += (self.rP_breve[-1] - self.rP_breve_old[-1]) - self.dt / self.tausyn * self.rP_breve_HI[-1]

		return self.rP_breve_HI


	def calc_dWPP_LO(self):
		# updates the low-passed update of WPP
		# Low-pass has the form d v_out =  dt/tau (v_in - v_out)

		for i in range(len(self.dWPP_LO)):
		      self.dWPP_LO[i] += self.dt / self.tausyn * (self.dWPP[i] - self.dWPP_LO[i])

		return self.dWPP_LO



	def calc_dendritic_updates(self, r0=None, u_tgt=None):

		"""
			this overwrites the dendritic updates by adding a low-pass on
			the interneuron voltages

		"""

		# calculate dendritic voltages from lookahead
		if r0 is not None:
			self.vbas[0] = self.WPP[0] @ self.r0

		for i in range(1, len(self.layers)-1):
			self.vbas[i] = self.calc_vbas(self.rP_breve[i-1], self.WPP[i])

		# add slow response to dendritic compartment of interneurons
		if self.inter_low_pass:
			self.vden[0] += self.dt / self.tausyn * (self.calc_vden(self.rP_breve[-2], self.WIP[-1]) - self.vden[0])
		else:
			# else, instant response
			self.vden[0] = self.calc_vden(self.rP_breve[-2], self.WIP[-1])

		for i in range(0, len(self.layers)-2):
			self.vapi[i] = self.calc_vapi(self.rP_breve[-1], self.BPP[i], self.rI_breve[-1], self.BPI[i])

		return self.vbas, self.vapi, self.vden

	def evolve_bw_synapses(self, active_bw_syn):
		# evolve the synapses in BPP of layer #active_bw_syn

		self.dBPP = [np.zeros(shape=BPP.shape) for BPP in self.BPP]

		if self.model == "LDRL":
			if self.pyr_hi_pass:
				self.dBPP[active_bw_syn] = self.dt * self.eta_bw[active_bw_syn] * np.outer(
					self.noise[active_bw_syn], self.rP_breve_HI[-1]
					)
				# add regularizer
				if self.gate_regularizer:
					self.dBPP[active_bw_syn] -= self.dt * self.alpha[active_bw_syn] * \
						self.eta_bw[active_bw_syn] * self.BPP[active_bw_syn] * self.d_activation[-1](self.rP_breve_HI[-1])
				else:
					self.dBPP[active_bw_syn] -= self.dt * self.alpha[active_bw_syn] * self.eta_bw[active_bw_syn] * self.BPP[active_bw_syn]

			else:
				self.dBPP[active_bw_syn] = self.dt * self.eta_bw[active_bw_syn] * np.outer(
					self.noise[active_bw_syn], self.rP_breve[-1]
					)
				# add regularizer
				if self.gate_regularizer:
					self.dBPP[active_bw_syn] -= self.dt * self.alpha[active_bw_syn] * \
						self.eta_bw[active_bw_syn] * self.BPP[active_bw_syn] * self.d_activation[-1](self.rP_breve[-1])
				else:
					self.dBPP[active_bw_syn] -= self.dt * self.alpha[active_bw_syn] * self.eta_bw[active_bw_syn] * self.BPP[active_bw_syn]
			


		elif self.model == "DTPDRL":
			if self.pyr_hi_pass:
				# self.dBPP[active_bw_syn] = - self.dt * self.eta_bw[active_bw_syn] * np.outer(
				# 	self.BPP[active_bw_syn]@self.rP_breve_HI[-1] + self.BPI[active_bw_syn]@self.rI_breve[-1] - self.noise[active_bw_syn],
				# 	self.rP_breve_HI[-1]
				# 	)
				self.dBPP[active_bw_syn] = - self.dt * self.eta_bw[active_bw_syn] * np.outer(
					self.BPP[active_bw_syn] @ self.rP_breve_HI[-1] - self.noise[active_bw_syn],
					self.rP_breve_HI[-1]
					)
			else:
				self.dBPP[active_bw_syn] = - self.dt * self.eta_bw[active_bw_syn] * np.outer(
					self.BPP[active_bw_syn]@self.rP_breve[-1] + self.BPI[active_bw_syn]@self.rI_breve[-1] - self.noise[active_bw_syn],
					self.rP_breve[-1]
					)
			# add regularizer
			self.dBPP[active_bw_syn] -= self.dt * self.alpha[active_bw_syn] * self.eta_bw[active_bw_syn] * self.BPP[active_bw_syn]


		return self.dBPP

















