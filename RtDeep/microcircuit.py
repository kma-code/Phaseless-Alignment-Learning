import numpy as np
import copy
#import matplotlib.pyplot as plt

# define activation functions
def linear(x):
	return x
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



class model:
	""" This class implements a generic microcircuit model """

	def __init__(self, dt, tauxi, Tpres, model, activation, d_activation, layers,
		uP_init, uI_init, WPP_init, WIP_init, BPP_init, BPI_init, gl, gden, gbas, gapi, gnI, gntgt):

		self.model = model # FA, BP or PBP
		self.layers = layers
		self.uP = copy.deepcopy(uP_init)
		self.uI = copy.deepcopy(uI_init)
		self.utgt = [np.zeros_like(uI_init[0])]
		# we also set up a buffer of voltages,
		# which corresponds to the value at the last time step
		self.uP_old = copy.deepcopy(self.uP)
		self.uI_old = copy.deepcopy(self.uI)

		self.activation = activation
		self.d_activation = d_activation

		# define the compartment voltages
		self.vbas = [np.zeros_like(uP) for uP in self.uP]
		self.vden = [np.zeros_like(uI) for uI in self.uI]
		self.vapi = [np.zeros_like(uP) for uP in self.uP[:-1]]

		self.WPP = WPP_init
		self.WIP = WIP_init
		self.BPP = BPP_init
		self.BPI = BPI_init

		self.gl = gl
		self.gden = gden
		self.gbas = gbas
		self.gapi = gapi
		self.gnI = gnI
		self.gntgt = gntgt

		self.Time = 0 # initialize a model timer
		self.dt = dt
		self.tauxi = tauxi
		self.Tpres = Tpres
		self.taueffP, self.taueffI = self.calc_taueff()

	def calc_taueff(self):
		# calculate tau_eff for pyramidals and interneuron
		# taueffP is one value per layer

		taueffP = []
		for i in self.uP:
			taueffP.append(1 / (self.gl + self.gbas + self.gapi))
		taueffP[-1] = 1 / (self.gl + self.gbas + self.gntgt)

		taueffI = [1 / (self.gl + self.gden + self.gnI)]

		return taueffP, taueffI


	def get_conductances(self):
		return self.gl, self.gden, self.gbas, self.gapi, self.gnI, self.gntgt


	def get_weights(self):
		return self.WPP, self.WIP, self.BPP, self.BPI


	def set_weights(self, WPP=None, WIP=None, BPP=None, BPI=None):
		if WPP != None: self.WPP = WPP
		if WIP != None: self.WIP = WIP
		if BPP != None: self.BPP = BPP
		if BPI != None: self.BPI = BPI


	def set_self_predicting_state(self):

		# set WIP and BIP to values corresponding to self-predicting state
		for i in range(len(self.layers)-1):
			self.BPI[i] = - self.BPP[i]
		self.WIP[-1] = self.gbas * (self.gl + self.gden) / (self.gden * (self.gl + self.gbas)) * self.WPP[-1]

	def get_voltages(self):
		return self.uP, self.uI


	def set_voltages(self, uP=None, uI=None):

		if uP != None:
			for i in range(len(self.layers)-1):
				print(i)
				self.uP[i] = uP[i]

		if uI != None:
			self.uI = uI

	def calc_vapi(self, uPvec, BPP_mat, uIvec, BPI_mat):
		# returns apical voltages in pyramidals of a given layer
		# input: uPvec: vector of pyramidal voltages in output layer
		# 		 WPP_mat: matrix connecting pyramidal to pyramidal
		# 		 uIvec: vector of interneuron voltages in output layer
		# 		 BPI_mat: matrix connecting interneurons to pyramidals

		return BPP_mat @ self.activation(uPvec) + BPI_mat @ self.activation(uIvec)

	def calc_vbas(self, uPvec, WPP_mat):
		# returns basal voltages in pyramidals of a given layer
		# input: uPvec: vector of pyramidal voltages in layer below
		# 		 WPP_mat: matrix connecting pyramidal to pyramidal

		return WPP_mat @ self.activation(uPvec)

	def calc_vden(self, uPvec, WIP_mat):
		# returns dendritic voltages in inteneurons
		# input: uPvec: vector of pyramidal voltages in layer below
		# 		 WIP_mat: matrix connecting pyramidal to pyramidal

		return WIP_mat @ self.activation(uPvec)


	def prospective_voltage(self, uvec, uvec_old, tau):
		# returns an approximation of the lookahead of voltage vector u at current time
		return uvec_old + tau * (uvec - uvec_old) / self.dt

	def evolve_voltages(self, r0=None):
		""" Evolves the pyramidal and interneuron voltages by one dt """
		""" using r0 as input rates """

		self.Time += self.dt

		# calculate lookahead
		self.uP_breve = [self.prospective_voltage(self.uP[i], self.uP_old[i], self.taueffP[i]) for i in range(len(self.uP))]
		self.uI_breve = [self.prospective_voltage(self.uI[i], self.uI_old[i], self.taueffI[i]) for i in range(len(self.uI))]

		self.uP_old = copy.deepcopy(self.uP)
		self.uI_old = copy.deepcopy(self.uI)

		# calculate dendritic voltages from lookahead
		if r0:
			self.vbas[0] = self.WPP[0] @ self.r0

		for i in range(1, len(self.layers)-1):
			self.vbas[i] = self.calc_vbas(self.uP_breve[i-1], self.WPP[i])

		self.vden[0] = self.calc_vden(self.uP_breve[-2], self.WIP[-1])

		for i in range(0, len(self.layers)-2):
			self.vapi[i] = self.calc_vapi(self.uP_breve[-1], self.BPP[i+1], self.uI_breve[-1], self.BPI[i+1])

		# update somatic potentials
		delta_uI = - self.gl * self.uI[-1] + self.gden * (self.vden[-1] - self.uI[-1])
		delta_uI += self.gnI * (self.uP[-1] - self.uI[-1])
		self.uI[-1] += self.dt * delta_uI

		for i in range(0, len(self.layers)-2):
			delta_uP = - self.gl * self.uP[i] + self.gbas * (self.vbas[i] - self.uP[i])
			delta_uP += self.gapi * (self.vapi[i] - self.uP[i])
			self.uP[i] += self.dt * delta_uP

		delta_uP = - self.gl * self.uP[-1] + self.gbas * (self.vbas[-1] - self.uP[-1])
		delta_uP += self.gntgt * (self.utgt[-1] - self.uP[-1])
		self.uP[-1] += self.dt * delta_uP 







