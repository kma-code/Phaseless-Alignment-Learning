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

# cosine similarity between tensors
def cos_sim(A, B):
	if A.ndim == 1 and B.ndim == 1:
		return A.T @ B / np.linalg.norm(A) / np.linalg.norm(B)
	else:
		return np.trace(A.T @ B) / np.linalg.norm(A) / np.linalg.norm(B)

def deepcopy_array(array):
	""" makes a deep copy of an array of np-arrays """
	out = [nparray.copy() for nparray in array]
	return out.copy()

def MSE(output, target):
	return (output - target).mean()


class base_model:
	""" This class implements a generic microcircuit model """

	def __init__(self, dt, Tpres, model, activation, d_activation, layers,
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

		# apply activation layer-wise,
		# so that we can easily disable it e.g. for last layer
		self.activation = [activation for layer in layers[1:-1]]
		self.activation.append(linear)
		self.d_activation = [d_activation for layer in layers[1:-1]]
		self.d_activation.append(d_linear)


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
		self.Time = np.round(self.Time + self.dt, decimals=np.int(-np.log10(self.dt)))

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
	def __init__(self, dt, dtxi, tausyn, Tbw, Tpres, noise_scale, alpha, inter_low_pass, pyr_hi_pass, model, activation, d_activation, layers,
					uP_init, uI_init, WPP_init, WIP_init, BPP_init, BPI_init,
					gl, gden, gbas, gapi, gnI, gntgt,
					eta_fw, eta_bw, eta_PI, eta_IP):

		# init base_model with same settings
		super().__init__(dt=dt, Tpres=Tpres,
			model=model, activation=activation, d_activation=d_activation, layers=layers,
            uP_init=uP_init, uI_init=uI_init,
            WPP_init=WPP_init, WIP_init=WIP_init, BPP_init=BPP_init, BPI_init=BPI_init,
            gl=gl, gden=gden, gbas=gbas, gapi=gapi, gnI=gnI, gntgt=gntgt,
            eta_fw=eta_fw, eta_bw=eta_bw, eta_PI=eta_PI, eta_IP=eta_IP)

		# new variables:

		# whether to low-pass filter the interneuron dendritic input
		self.inter_low_pass = inter_low_pass
		self.pyr_hi_pass = pyr_hi_pass
		# noise time scale
		self.dtxi = dtxi
		# decimals of dt
		self.dt_decimals = np.int(-np.log10(self.dt))
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
		self.noise_breve = [np.zeros(shape=uP.shape) for uP in self.uP]

		# init a counter for time steps after which to resample noise
		self.noise_counter = 0
		self.noise_total_counts = np.round(self.dtxi / self.dt, decimals=self.dt_decimals)

		# init a high-pass filtered version of rP_breve
		self.rP_breve_HI = deepcopy_array(self.rP_breve)

		# regularizer for backward weights
		self.alpha = alpha


	def evolve_system(self, r0=None, u_tgt=None, learn_weights=True, learn_bw_weights=True):

		""" 
			This overwrites the vanilla system evolution and implements
			phased noise
		"""

		# update which backwards weights to learn
		if learn_bw_weights and self.Time % self.Tbw == 0:
			print(f"Current time: {self.Time}s")
			self.active_bw_syn = 0 if self.active_bw_syn == len(self.BPP) - 1 else self.active_bw_syn + 1
			print(f"Learning backward weights to layer {self.active_bw_syn + 1}")
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
			for i in range(len(self.dWPP)):
				self.WPP[i] += self.dWPP[i]
			for i in range(len(self.dWIP)):
				self.WIP[i] += self.dWIP[i]
			for i in range(len(self.dBPI)):
				self.BPI[i] += self.dBPI[i]
		if learn_bw_weights:
			for i in range(len(self.dBPP)):
				self.BPP[i] += self.dBPP[i]

		# increase timer
		self.Time = np.round(self.Time + self.dt, decimals=self.dt_decimals)


	def evolve_voltages(self, r0=None, u_tgt=None, inject_noise=False):
		""" 
			Overwrites voltage evolution:
			Evolves the pyramidal and interneuron voltages by one dt
			using r0 as input rates
			>> Injects noise into vapi
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

		self.duP, self.duI = self.calc_somatic_updates(u_tgt)

		return self.duP, self.duI


	def inject_noise(self, layer, noise_scale):

		"""
			 this function injects noise into a given layer
			 by adding it to the apical potential
		"""

		# save current noise for noise_breve
		self.noise_old = deepcopy_array(self.noise)

		# if dtxi timesteps have passed, sample new noise
		if np.all(self.noise[layer] == 0) or self.noise_counter % self.noise_total_counts == 0:

			stdev = noise_scale[layer] * np.max(np.abs(self.uP[layer]))
			self.noise[layer] = np.random.normal(loc=0, scale=stdev, size=self.vapi[layer].shape)
			self.noise_counter = 0

		self.noise_counter += 1

		self.vapi[layer] += self.noise[layer]
		self.noise_breve[layer] = self.prospective_voltage(self.noise[layer], self.noise_old[layer], self.taueffP[layer])

	def calc_rP_breve_HI(self):
		# updates the high-passed instantaneous rate rP_breve_HI which is used to update BPP
		# High-pass has the form d v_out = d v_in - dt/tau * v_out

		# we will only need rP_breve_HI coming from the final layer, so we freeze the others
		# for i in range(len(self.rP_breve)):
		#       self.rP_breve_HI[i] += (self.rP_breve[i] - self.rP_breve_old[i]) - self.dt / self.tausyn * self.rP_breve_HI[i]
		self.rP_breve_HI[-1] += (self.rP_breve[-1] - self.rP_breve_old[-1]) - self.dt / self.tausyn * self.rP_breve_HI[-1]

		return self.rP_breve_HI



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
			else:
				self.dBPP[active_bw_syn] = self.dt * self.eta_bw[active_bw_syn] * np.outer(
					self.noise[active_bw_syn], self.rP_breve[-1]
					)
			# add regularizer
			self.dBPP[active_bw_syn] -= self.dt * self.alpha[active_bw_syn] * self.eta_bw[active_bw_syn] * self.BPP[active_bw_syn]

		elif self.model == "DTPDRL":
			if self.pyr_hi_pass:
				self.dBPP[active_bw_syn] = - self.dt * self.eta_bw[active_bw_syn] * np.outer(
					self.BPP[active_bw_syn]@self.rP_breve_HI[-1]-self.BPI[active_bw_syn]@self.rI_breve[-1]-self.noise[active_bw_syn],
					self.rP_breve_HI[-1]
					)
			else:
				self.dBPP[active_bw_syn] = - self.dt * self.eta_bw[active_bw_syn] * np.outer(
					self.BPP[active_bw_syn]@self.rP_breve[-1]-self.BPI[active_bw_syn]@self.rI_breve[-1]-self.noise[active_bw_syn],
					self.rP_breve[-1]
					)
			# add regularizer
			self.dBPP[active_bw_syn] -= self.dt * self.alpha[active_bw_syn] * self.eta_bw[active_bw_syn] * self.BPP[active_bw_syn]


		return self.dBPP


# low pass filter adapted from user3123955 and Warren Weckesser @ SO
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

















