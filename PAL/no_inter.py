from microcircuit import *

# this module implements a new class where the interneuron is replaced by local activity

class no_inter_model(phased_noise_model):
	""" 
		This class inherits all properties from the phased_noise_model class
		and removes the interneuron, replacing its signal in hidden layers by u_i.
	"""
	def __init__(self, dt, dtxi, tausyn, Tbw, Tpres, noise_scale, alpha, inter_low_pass, pyr_hi_pass, dWPP_low_pass, gate_regularizer, noise_mode,
					model, activation, layers,
					uP_init, uI_init, WPP_init, WIP_init, BPP_init, BPI_init,
					gl, gden, gbas, gapi, gnI, gntgt,
					eta_fw, eta_bw, eta_PI, eta_IP):

		# init base_model with same settings
		super().__init__(dt=dt, dtxi=dtxi, tausyn=tausyn, Tbw=Tbw, Tpres=Tpres, noise_scale=noise_scale, alpha=alpha,
			inter_low_pass=inter_low_pass, pyr_hi_pass=pyr_hi_pass, dWPP_low_pass=dWPP_low_pass, gate_regularizer=gate_regularizer, noise_mode=noise_mode,
			model=model, activation=activation, layers=layers,
            uP_init=uP_init, uI_init=uI_init,
            WPP_init=WPP_init, WIP_init=WIP_init, BPP_init=BPP_init, BPI_init=BPI_init,
            gl=gl, gden=gden, gbas=gbas, gapi=gapi, gnI=gnI, gntgt=gntgt,
            eta_fw=eta_fw, eta_bw=eta_bw, eta_PI=eta_PI, eta_IP=eta_IP)



	def calc_vapi(self, rPvec, BPP_mat, uP_breve):
		# overwrites the usual calculation using pyr + inter

		# returns apical voltages in pyramidals of a given layer
		# input: rPvec: vector of rates from pyramidal voltages in output layer
		# 		 WPP_mat: matrix connecting pyramidal to pyramidal
		# 		 uP_breve: instantaneos voltages of current hidden layer

		return BPP_mat @ rPvec - uP_breve


	def evolve_system(self, r0=None, u_tgt=None, learn_weights=True, learn_bw_weights=True):

		""" 
			This takes the phased_noise model and skips all calculations for interneurons
		"""

		# update which backwards weights to learn
		if learn_bw_weights and self.Time % self.Tbw == 0:
			# print(f"Current time: {self.Time}s")
			self.active_bw_syn = 0 if self.active_bw_syn == len(self.BPP) - 1 else self.active_bw_syn + 1
			# print(f"Learning backward weights to layer {self.active_bw_syn + 1}")
			self.noise_counter = 0

		# calculate voltage evolution, including low pass on interneuron synapses
		# see calc_dendritic updates below
		self.duP, _ = self.evolve_voltages(r0, u_tgt, inject_noise=learn_bw_weights) # includes recalc of rP_breve

		# calculate hi-passed rP_breve for synapse BPP
		self.rP_breve_HI = self.calc_rP_breve_HI()

		if learn_weights:
			self.dWPP, _, _, _ = self.evolve_synapses(r0)
		if learn_bw_weights:
			self.dBPP = self.evolve_bw_synapses(self.active_bw_syn)

		# apply evolution
		for i in range(len(self.duP)):
			self.uP[i] += self.duP[i]

		if learn_weights:
			if self.dWPP_low_pass:
				# calculate lo-passed update of WPP
				self.dWPP_LO = self.calc_dWPP_LO()
				for i in range(len(self.dWPP_LO)):
					self.WPP[i] += self.dWPP_LO[i]
			else:
				for i in range(len(self.dWPP)):
					self.WPP[i] += self.dWPP[i]
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
			This takes the phased_noise model and skips all calculations for interneurons
		"""

		self.duP = [np.zeros(shape=uP.shape) for uP in self.uP]
		# self.duI = [np.zeros(shape=uI.shape) for uI in self.uI]

		# same for dendritic voltages and rates
		self.rP_breve_old = deepcopy_array(self.rP_breve)
		# self.rI_breve_old = deepcopy_array(self.rI_breve)
		if self.r0 is not None:
			self.r0_old = self.r0.copy()
		self.vbas_old = deepcopy_array(self.vbas)
		self.vden_old = deepcopy_array(self.vden)
		self.vapi_old = deepcopy_array(self.vapi)

		# calculate lookahead
		self.uP_breve = [self.prospective_voltage(self.uP[i], self.uP_old[i], self.taueffP[i]) for i in range(len(self.uP))]
		# self.uI_breve = [self.prospective_voltage(self.uI[i], self.uI_old[i], self.taueffI[i]) for i in range(len(self.uI))]
		# calculate rate of lookahead: phi(ubreve)
		self.rP_breve = [self.activation[i](self.uP_breve[i]) for i in range(len(self.uP_breve))]
		# self.rI_breve = [self.activation[-1](self.uI_breve[-1])]
		self.r0 = r0

		# before modifying uP and uI, we need to save copies
		# for future calculation of u_breve
		self.uP_old = deepcopy_array(self.uP)
		# self.uI_old = deepcopy_array(self.uI)

		self.vbas, self.vapi, _ = self.calc_dendritic_updates(r0, u_tgt)

		# inject noise into newly calculated vapi before calculating update du
		if inject_noise: self.inject_noise(layer=self.active_bw_syn, noise_scale=self.noise_scale)

		self.duP, _ = self.calc_somatic_updates(u_tgt, inject_noise=inject_noise)

		return self.duP, None


	def calc_dendritic_updates(self, r0=None, u_tgt=None):

		"""
			this overwrites the dendritic updates of phased_model
			by leaving out the interneuron and using uP_breve instead

		"""

		# calculate dendritic voltages from lookahead
		if r0 is not None:
			self.vbas[0] = self.WPP[0] @ self.r0

		for i in range(1, len(self.layers)-1):
			self.vbas[i] = self.calc_vbas(self.rP_breve[i-1], self.WPP[i])

		# calculate vapi using uP_breve
		for i in range(0, len(self.layers)-2):
			self.vapi[i] = self.calc_vapi(self.rP_breve[-1], self.BPP[i], self.uP_breve[i])

		return self.vbas, self.vapi, None

	def calc_somatic_updates(self, u_tgt=None, inject_noise=False):
		"""
			this overwrites the somatic update rules
			only difference to phased_model: doesn't update interneurons

		"""

		# update somatic potentials
		# ueffI = self.taueffI[-1] * (self.gden * self.vden[-1] + self.gnI * self.uP_breve[-1])
		# delta_uI = (ueffI - self.uI[-1]) / self.taueffI[-1]
		# self.duI[-1] = self.dt * delta_uI

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

		return self.duP, None


	def evolve_synapses(self, r0):

		# this overwrites the base model's synaptic evolution
		# difference: removes all plasticity of BPI and WIP
		
		""" evolves all synapses by a dt """


		"""
			plasticity of WPP

		"""

		self.dWPP = [np.zeros(shape=WPP.shape) for WPP in self.WPP]
		# self.dWIP = [np.zeros(shape=WIP.shape) for WIP in self.WIP]
		self.dBPP = [np.zeros(shape=BPP.shape) for BPP in self.BPP]
		# self.dBPI = [np.zeros(shape=BPI.shape) for BPI in self.BPI]

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

		# self.dWIP[-1] = self.dt * self.eta_IP[-1] * np.outer(
		# 			self.rI_breve[-1] - self.activation[-1](self.gden / (self.gl + self.gden) * self.vden_old[-1]),
		# 											self.rP_breve_old[-2])

		"""
			plasticity of BPI

		"""

		# for i in range(0, len(self.BPI)):
		# 	if self.eta_PI[i] != 0:
		# 		self.dBPI[i] = self.dt * self.eta_PI[i] * np.outer(-self.vapi[i], self.rI_breve_old[-1])

		"""
			plasticity of BPP

		"""

		if self.model == 'FA':
			# do nothing
			0

		elif self.model == 'BP':
			self.set_weights(BPP = [WPP.T for WPP in self.WPP[1:]])


		return self.dWPP, None, self.dBPP, None











