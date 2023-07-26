#!/usr/bin/env python
# PyTorch implementation of the Latent Equilibrium model for different layers.

# Authors: Benjamin Ellenberger (benelot@github)

from datetime import datetime

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.layered_torch_utils import set_tensor
from model.network_params import ModelVariant, TargetType
from utils.torch_utils import SimpleDataset

# give this to each dataloader
def dataloader_seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Conv2d(object):
    def __init__(self, num_channels, num_filters, kernel_size, batch_size, input_size, act_function, padding=0, stride=1, learning_rate=0.1):
        self.input_size = input_size
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.target_size = (np.floor((self.input_size + (2 * self.padding) - self.kernel_size) / self.stride) + 1).astype(int)

        self.tau = 10.0
        self.dt = 0.1
        self.learning_rate_W = learning_rate
        self.learning_rate_biases = learning_rate

        self.act_function = act_function.f
        self.act_func_deriv = act_function.df

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_W = True

        self.kernel = torch.empty(self.num_filters, self.num_channels, self.kernel_size, self.kernel_size).normal_(mean=0, std=0.05).to(self.device)
        self.biases = torch.empty(self.num_filters).normal_(mean=0, std=0.05).to(self.device)

        self.unfold = nn.Unfold(kernel_size=(self.kernel_size, self.kernel_size),
                                padding=self.padding,
                                stride=self.stride).to(self.device)
        self.fold = nn.Fold(output_size=(self.input_size, self.input_size),
                            kernel_size=(self.kernel_size, self.kernel_size),
                            padding=self.padding,
                            stride=self.stride).to(self.device)

        self.voltages = torch.zeros([1, self.num_filters, self.target_size, self.target_size], device=self.device)
        self.voltages_deriv = None
        self.voltage_lookaheads = torch.zeros([1, self.num_filters, self.target_size], device=self.device)
        self.basal_inputs = None
        self.errors = torch.zeros([1, self.num_filters, self.target_size, self.target_size], device=self.device)

        self.rho = None
        self.rho_flat = None
        self.rho_deriv = torch.zeros([1, self.num_filters, self.target_size, self.target_size], device=self.device)

    def train(self):
        """
        Enables training.
        :return:
        """
        self.train_W = True

    def eval(self):
        """
        Disables training.
        :return:
        """
        self.train_W = False

    def _adapt_parallel_network_qty(self):
        """Adapt number of voltage sets to batch size (if more sets are required, repeat current sets, if less are required drop current sets).
        Returns:

        """
        batch_size = self.rho_flat.shape[0]
        if len(self.voltages.shape) != 3 or self.voltages.shape[0] != batch_size:
            voltage_size = self.voltages.shape[0]
            repeats = int(batch_size / voltage_size)
            remainder = batch_size % voltage_size
            repetition_vector = torch.tensor([repeats], device=self.device).repeat(voltage_size)
            repetition_vector[-1] = repetition_vector[-1] + remainder

            self.voltages = torch.repeat_interleave(self.voltages, repetition_vector, dim=0).clone()
            self.errors = torch.repeat_interleave(self.errors, repetition_vector, dim=0).clone()
            self.voltage_lookaheads = torch.repeat_interleave(self.voltage_lookaheads, repetition_vector, dim=0).clone()

    def forward(self, rho, rho_deriv):
        self.rho_flat = self.unfold(rho.clone())
        self.weights_flat = self.kernel.reshape(self.num_filters, -1)
        # self.biases_flat = self.kernel_biases.reshape(self.num_filters, -1)

        self._adapt_parallel_network_qty()

        self.basal_inputs = self.weights_flat @ self.rho_flat
        self.basal_inputs = self.basal_inputs.reshape(self.batch_size, self.num_filters, self.target_size, self.target_size) + self.biases.reshape(1, -1, 1, 1)

        self.voltages_deriv = 1.0 / self.tau * (self.basal_inputs - self.voltages + self.errors)
        self.voltage_lookaheads = self.voltages + self.tau * self.voltages_deriv
        self.voltages = self.voltages + self.dt * self.voltages_deriv

        self.rho = self.act_function(self.voltage_lookaheads)
        self.rho_deriv = self.act_func_deriv(self.voltage_lookaheads)

        self.errors = self._calculate_errors(self.voltage_lookaheads, rho_deriv, self.basal_inputs)
        return self.rho, self.rho_deriv

    def update_weights(self, errors, with_optimizer=False):
        previous_layer_errors = self.errors

        self.errors = errors
        if self.train_W:

            dot_weights = self.get_weight_derivatives()
            self.kernel.grad = -self.dt * dot_weights  # our gradients are inverse to pytorch gradients

            dot_biases = self.get_bias_derivatives()  # do bias update
            self.biases.grad = -self.dt * dot_biases  # our gradients are inverse to pytorch gradients

            if not with_optimizer:  # minus because pytorch gradients are inverse to our gradients
                self.kernel -= self.kernel.grad * self.learning_rate_W
                self.biases -= self.biases.grad * self.learning_rate_biases

        return previous_layer_errors

    # ### CALCULATE WEIGHT DERIVATIVES ### #

    def get_weight_derivatives(self):
        """
        Return weight derivative calculated from current rate and errors.
        Args:

        Returns:
            weight_derivative: e * r^T * η * weight_mask

        """
        e = self.errors.reshape(self.batch_size, self.num_filters, -1)
        dW = e @ self.rho_flat.permute(0, 2, 1)
        dW = dW.mean(dim=0)
        dW = dW.reshape((self.num_filters, self.num_channels, self.kernel_size, self.kernel_size))
        return dW

    # ### CALCULATE BIAS DERIVATIVES ### #

    def get_bias_derivatives(self):
        """
        Return bias derivative.
        Args:

        Returns:
            bias_derivative: e * η * bias_mask

        """
        return self.errors.mean([0, 2, 3])

    def _calculate_errors(self, voltage_lookaheads, rho_deriv, basal_inputs):
        """
        Calculate:
            layerwise error:    e = diag(r') W^T (U - Wr)

        Args:
            voltage_lookaheads:
            rho_deriv:
            basal_inputs:

        Returns:
            errors:

        """
        # e
        e = (voltage_lookaheads - basal_inputs).reshape(self.batch_size, self.num_filters, -1)
        err = self.weights_flat.T @ e
        err = self.fold(err)
        err = rho_deriv * err
        return err

    def save_layer(self, logdir, i):
        np.save(logdir + "/layer_" + str(i) + "_weights.npy", self.kernel.detach().cpu().numpy())

    def load_layer(self, logdir, i):
        kernel = np.load(logdir + "/layer_" + str(i) + "_weights.npy")
        self.kernel = set_tensor(torch.from_numpy(kernel))

    def __call__(self, rho, rho_deriv):
        return self.forward(rho, rho_deriv)

    def parameters(self):
        return [self.kernel, self.biases]


class MaxPool2d(object):
    # These are not really neurons...
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        self.target_size = kernel_size

    def train(self):
        """
        Enables training.
        :return:
        """
        pass

    def eval(self):
        """
        Disables training.
        :return:
        """
        pass

    def forward(self, rho, rho_deriv):
        rho_out, self.idxs_rho = F.max_pool2d(rho, self.kernel_size, return_indices=True)
        rho_deriv_out = self._maxpool2d_with_indices(rho_deriv, self.idxs_rho)
        return rho_out, rho_deriv_out

    def _maxpool2d_with_indices(self, t, indices):
        flattened_tensor = t.flatten(start_dim=2)
        output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
        return output

    def update_weights(self, errors, with_optimizer=False):
        return F.max_unpool2d(errors, self.idxs_rho, self.kernel_size)

    def save_layer(self, logdir, i):
        pass

    def load_layer(self, logdir, i):
        pass

    def __call__(self, rho, rho_deriv):
        return self.forward(rho, rho_deriv)

    def parameters(self):
        return []


class AvgPool2d(object):
    # These are not really neurons...
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        self.target_size = kernel_size

    def train(self):
        """
        Enables training.
        :return:
        """
        pass

    def eval(self):
        """
        Disables training.
        :return:
        """
        pass

    def forward(self, rho, rho_deriv):
        return F.avg_pool2d(rho, self.kernel_size), F.avg_pool2d(rho_deriv, self.kernel_size)

    def update_weights(self, errors, with_optimizer=False):
        return F.interpolate(errors, scale_factor=(1, 1, self.kernel_size,self.kernel_size))

    def save_layer(self, logdir, i):
        pass

    def load_layer(self, logdir, i):
        pass

    def __call__(self, rho, rho_deriv):
        return self.forward(rho, rho_deriv)

    def paremeters(self):
        return []


class Projection(object):
    def __init__(self, input_size, target_size, act_function, dtype=torch.float32):
        self.input_size = input_size
        self.B, self.C, self.H, self.W = self.input_size
        self.target_size = target_size
        self.act_function = act_function.f
        self.act_func_deriv = act_function.df

        self.tau = 10.0
        self.dt = 0.1
        self.learning_rate_W = 0
        self.learning_rate_biases = 0

        self.dtype = dtype
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_W = True

        self.Hid = self.C * self.H * self.W
        self.weights = torch.empty((self.Hid, self.target_size)).normal_(mean=0.0, std=0.05).to(self.device)
        self.biases = torch.empty(self.target_size).normal_(mean=0.0, std=0.05).to(self.device)

        self.voltages = torch.zeros([1, self.target_size], device=self.device)
        self.voltages_deriv = None
        self.voltage_lookaheads = torch.zeros([1, self.target_size], device=self.device)
        self.basal_inputs = None
        self.errors = torch.zeros([1, self.target_size], device=self.device)

        self.rho = None
        self.rho_input = None
        self.rho_deriv = torch.zeros([1, self.target_size], dtype=dtype, device=self.device)

    def train(self):
        """
        Enables training.
        :return:
        """
        self.train_W = True

    def eval(self):
        """
        Disables training.
        :return:
        """
        self.train_W = False

    def _adapt_parallel_network_qty(self):
        """Adapt number of voltage sets to batch size (if more sets are required, repeat current sets, if less are required drop current sets).
        Returns:

        """
        batch_size = self.rho_input.shape[0]
        if len(self.voltages.shape) != 2 or self.voltages.shape[0] != batch_size:
            voltage_size = self.voltages.shape[0]
            repeats = int(batch_size / voltage_size)
            remainder = batch_size % voltage_size
            repetition_vector = torch.tensor([repeats], device=self.device).repeat(voltage_size)
            repetition_vector[-1] = repetition_vector[-1] + remainder

            self.voltages = torch.repeat_interleave(self.voltages, repetition_vector, dim=0).clone()
            self.errors = torch.repeat_interleave(self.errors, repetition_vector, dim=0).clone()
            self.voltage_lookaheads = torch.repeat_interleave(self.voltage_lookaheads, repetition_vector, dim=0).clone()

    def forward(self, rho, rho_deriv):
        self.rho_input = rho.clone()
        rho = rho.reshape((len(rho), -1))

        self._adapt_parallel_network_qty()

        self.basal_inputs = torch.matmul(rho, self.weights) + self.biases

        self.voltages_deriv = 1.0 / self.tau * (self.basal_inputs - self.voltages + self.errors)
        self.voltage_lookaheads = self.voltages + self.tau * self.voltages_deriv
        self.voltages = self.voltages + self.dt * self.voltages_deriv

        self.rho = self.act_function(self.voltage_lookaheads)
        self.rho_deriv = self.act_func_deriv(self.voltage_lookaheads)

        self.errors = self._calculate_errors(self.voltage_lookaheads, rho_deriv, self.basal_inputs)
        return self.rho, self.rho_deriv

    def update_weights(self, errors, with_optimizer=False):
        previous_layer_errors = self.errors

        self.errors = errors
        if self.train_W:

            dot_weights = self.get_weight_derivatives()
            self.weights.grad = -self.dt * dot_weights  # our gradients are inverse to pytorch gradients

            dot_biases = self.get_bias_derivatives()  # do bias update
            self.biases.grad = -self.dt * dot_biases # our gradients are inverse to pytorch gradients

            if not with_optimizer:  # minus because pytorch gradients are inverse to our gradients
                self.weights -= self.weights.grad * self.learning_rate_W
                self.biases -= self.biases.grad * self.learning_rate_biases

        return previous_layer_errors

    # ### CALCULATE WEIGHT DERIVATIVES ### #

    def get_weight_derivatives(self):
        """
        Return weight derivative calculated from current rate and errors.
        Args:

        Returns:
            weight_derivative: e * r^T * η * weight_mask

        """
        # If the input is served as a single sample, it is not in batch form, but this here requires it.
        rho_input_flat = self.rho_input.reshape((len(self.rho_input), -1))
        return (torch.einsum('bi,bj->bij', rho_input_flat, self.errors)).mean(0)

    # ### CALCULATE BIAS DERIVATIVES ### #

    def get_bias_derivatives(self):
        """
        Return bias derivative.
        Args:

        Returns:
            bias_derivative: e * η * bias_mask

        """
        return self.errors.mean(0)

    def _calculate_errors(self, voltage_lookaheads, rho_deriv, basal_inputs):
        """
        Calculate:
            layerwise error:    e = diag(r') W^T (U - Wr)

        Args:
            voltage_lookaheads:
            rho_deriv:
            basal_inputs:

        Returns:
            errors:

        """
        # e
        err = torch.matmul(voltage_lookaheads - basal_inputs, self.weights.t())
        err = err.reshape((len(err), self.C, self.H, self.W))
        err = rho_deriv * err
        return err

    def save_layer(self, logdir, i):
        np.save(logdir + "/layer_" + str(i) + "_weights.npy", self.weights.detach().cpu().numpy())

    def load_layer(self, logdir, i):
        weights = np.load(logdir + "/layer_" + str(i) + "_weights.npy")
        self.weights = set_tensor(torch.from_numpy(weights))

    def __call__(self, rho, rho_deriv):
        return self.forward(rho, rho_deriv)

    def parameters(self):
        return [self.weights, self.biases]


class Linear(object):
    def __init__(self, input_size, target_size, act_function, algorithm='BP', dtype=torch.float32, learning_rate=0.1, wn_sigma=0):
        self.input_size = input_size
        self.target_size = target_size
        self.act_function = act_function.f
        self.act_func_deriv = act_function.df

        self.tau = 10.0
        self.dt = 0.1
        self.learning_rate_W = learning_rate
        self.learning_rate_biases = learning_rate

        self.dtype = dtype
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_W = True

        self.algorithm = algorithm
        self.wn_sigma = wn_sigma

        # initialize weights, voltages, rates, errors
        self.weights = torch.empty([self.input_size, self.target_size]).normal_(mean=0.0, std=0.05).to(self.device)
        if self.algorithm == 'FA':
            self.bw_weights = torch.empty([self.input_size, self.target_size]).T.normal_(mean=0.0, std=0.05).to(self.device)
        self.biases = torch.empty(self.target_size).normal_(mean=0.0, std=0.05).to(self.device)

        self.voltages = torch.zeros([1, self.target_size], device=self.device)
        self.voltages_deriv = None
        self.voltage_lookaheads = torch.zeros([1, self.target_size], device=self.device)
        self.basal_inputs = None
        self.errors = torch.zeros([1, self.target_size], device=self.device)

        self.rho = None
        self.rho_input = None
        self.rho_deriv = torch.zeros([1, self.target_size], dtype=dtype, device=self.device)

    def train(self):
        """
        Enables training.
        :return:
        """
        self.train_W = True

    def eval(self):
        """
        Disables training.
        :return:
        """
        self.train_W = False

    def _adapt_parallel_network_qty(self):
        """Adapt number of voltage sets to batch size (if more sets are required, repeat current sets, if less are required drop current sets).
        Returns:

        """
        batch_size = self.rho_input.shape[0]
        if len(self.voltages.shape) != 2 or self.voltages.shape[0] != batch_size:
            voltage_size = self.voltages.shape[0]
            repeats = int(batch_size / voltage_size)
            remainder = batch_size % voltage_size
            repetition_vector = torch.tensor([repeats], device=self.device).repeat(voltage_size)
            repetition_vector[-1] = repetition_vector[-1] + remainder

            self.voltages = torch.repeat_interleave(self.voltages, repetition_vector, dim=0).clone()
            self.errors = torch.repeat_interleave(self.errors, repetition_vector, dim=0).clone()
            self.voltage_lookaheads = torch.repeat_interleave(self.voltage_lookaheads, repetition_vector, dim=0).clone()

    def forward(self, rho, rho_deriv):
        self.rho_input = rho.clone()

        self._adapt_parallel_network_qty()

        self.basal_inputs = torch.matmul(rho, self.weights) + self.biases

        # voltages are basal + error + noise
        self.voltages_deriv = 1.0 / self.tau * (self.basal_inputs - self.voltages + self.errors + self.wn_sigma * torch.randn(self.voltages.size(), device=self.device))
        self.voltage_lookaheads = self.voltages + self.tau * self.voltages_deriv
        self.voltages = self.voltages + self.dt * self.voltages_deriv

        self.rho = self.act_function(self.voltage_lookaheads)
        self.rho_deriv = self.act_func_deriv(self.voltage_lookaheads)

        self.errors = self._calculate_errors(self.voltage_lookaheads, rho_deriv, self.basal_inputs)
        return self.rho, self.rho_deriv

    def update_weights(self, errors, with_optimizer=False):
        previous_layer_errors = self.errors

        self.errors = errors
        if self.train_W:

            dot_weights = self.get_weight_derivatives()
            self.weights.grad = -self.dt * dot_weights  # our gradients are inverse to pytorch gradients

            dot_biases = self.get_bias_derivatives()  # do bias update
            self.biases.grad = -self.dt * dot_biases  # our gradients are inverse to pytorch gradients

            if not with_optimizer:  # minus because pytorch gradients are inverse to our gradients
                self.weights -= self.weights.grad * self.learning_rate_W
                self.biases -= self.biases.grad * self.learning_rate_biases

        return previous_layer_errors

    # ### CALCULATE WEIGHT DERIVATIVES ### #

    def get_weight_derivatives(self):
        """
        Return weight derivative calculated from current rate and errors.
        Args:

        Returns:
            weight_derivative: e * r^T * η * weight_mask

        """
        # If the input is served as a single sample, it is not in batch form, but this here requires it.
        if len(self.rho_input.shape) == 1:
            self.rho_input = self.rho_input.unsqueeze(0)
        return (torch.einsum('bi,bj->bij', self.rho_input, self.errors)).mean(0)

    # ### CALCULATE BIAS DERIVATIVES ### #

    def get_bias_derivatives(self):
        """
        Return bias derivative.
        Args:

        Returns:
            bias_derivative: e * η * bias_mask

        """
        return self.errors.mean(0)

    def _calculate_errors(self, voltage_lookaheads, rho_deriv, basal_inputs):
        """
        Calculate:
            layerwise error:    e = diag(r') W^T (U - Wr)

        Args:
            voltage_lookaheads:
            rho_deriv:
            basal_inputs:

        Returns:
            errors:

        """
        # e
        if self.algorithm == 'BP':
            err = rho_deriv * torch.matmul(voltage_lookaheads - basal_inputs, self.weights.t())
        elif self.algorithm == 'FA':
            err = rho_deriv * torch.matmul(voltage_lookaheads - basal_inputs, self.bw_weights)

        return err

    def save_layer(self, logdir, i):
        np.save(logdir + "/layer_" + str(i) + "_weights.npy", self.weights.detach().cpu().numpy())
        np.save(logdir + "/layer_" + str(i) + "_biases.npy", self.biases.detach().cpu().numpy())

    def load_layer(self, logdir, i):
        weights = np.load(logdir + "/layer_" + str(i) + "_weights.npy")
        self.weights = torch.from_numpy(weights).float().to(self.device)
        biases = np.load(logdir + "/layer_" + str(i) + "_biases.npy")
        self.biases = torch.from_numpy(biases).float().to(self.device)

    def __call__(self, rho, rho_deriv):
        return self.forward(rho, rho_deriv)

    def parameters(self):
        return [self.weights, self.biases]


class Linear_PAL(Linear):
    """
        This class inherits all properties from the parent class 'Linear' and adds the function to add noise

        We need to modify the forward and backward passes,
        and inject noise. 

    """
    def __init__(self, input_size, target_size, act_function, algorithm='PAL', dtype=torch.float32, learning_rate=0.1, learning_rate_bw=1.0, regularizer=1e-3, tau_xi=1.0, tau_HP=1.0, tau_LO=1000.0, sigma=1e-2, wn_sigma=0.0):

        # init parent class with same settings
        super().__init__(input_size=input_size, target_size=target_size, act_function=act_function, algorithm=algorithm, dtype=torch.float32, learning_rate=learning_rate, wn_sigma=wn_sigma)


        if self.algorithm != 'PAL':
            raise Exception("Unknown algorithm: this linear layer is only implemented for PAL")

        self.train_B = True

        self.learning_rate_B = learning_rate_bw
        self.regularizer = regularizer

        self.tau_xi = tau_xi    # time constant of OU noise
        self.tau_HP = tau_HP    # time constant of high-pass filter
        self.tau_LO = tau_LO    # time constant of low-pass filter

        self.sigma = sigma      # scale of injected noise

        self.bw_weights = torch.empty([self.input_size, self.target_size]).T.normal_(mean=0.0, std=0.05).to(self.device)
        self.noise = torch.zeros([1, self.target_size], device=self.device)
        self.rho_HP = torch.zeros([1, self.target_size], device=self.device)
        self.Delta_rho = torch.zeros([1, self.target_size], device=self.device)

    def train(self):
        """
        Enables training.
        :return:
        """
        self.train_W = True
        self.train_B = True

    def eval(self):
        """
        Disables training.
        :return:
        """
        self.train_W = False
        self.train_B = False

    def forward(self, rho, rho_deriv):
        self.rho_input = rho.clone()

        self._adapt_parallel_network_qty()

        self.basal_inputs = torch.matmul(rho, self.weights) + self.biases

        # calculate new noise
        self.noise = self._update_OU_noise(self.noise)
        # voltage is basal + error + OU-noise + white noise
        self.voltages_deriv = 1.0 / self.tau * (self.basal_inputs - self.voltages + self.errors + self.noise + self.wn_sigma * torch.randn(self.voltages.size(), device=self.device))
        self.voltage_lookaheads = self.voltages + self.tau * self.voltages_deriv
        self.voltages = self.voltages + self.dt * self.voltages_deriv

        # before updating rho, calculate Delta (needed for high-pass)
        # print("a", self.act_function(self.voltage_lookaheads).size())
        if self.rho is not None:
            self.Delta_rho = self.act_function(self.voltage_lookaheads) - self.rho
        # print(self.Delta_rho.size())

        self.rho = self.act_function(self.voltage_lookaheads)
        self.rho_deriv = self.act_func_deriv(self.voltage_lookaheads)

        # calculate high-pass of current layer rate, to be passed to layer below
        self.rho_HP = self._calc_HP(Delta_rho=self.Delta_rho)

        self.errors = self._calculate_errors(self.voltage_lookaheads, rho_deriv, self.basal_inputs)

        return self.rho, self.rho_deriv, self.rho_HP, self.noise

    def _calculate_errors(self, voltage_lookaheads, rho_deriv, basal_inputs):
        """
        Calculate:
            layerwise error:    e = diag(r') B (U - Wr)

        Args:
            voltage_lookaheads:
            rho_deriv:
            basal_inputs:

        Returns:
            errors:

        """
        # e
        err = rho_deriv * torch.matmul(voltage_lookaheads - basal_inputs, self.bw_weights)

        return err

    def _adapt_parallel_network_qty(self):
        """Adapt number of voltage sets to batch size (if more sets are required, repeat current sets, if less are required drop current sets).
        Returns:

        """
        batch_size = self.rho_input.shape[0]
        if len(self.voltages.shape) != 2 or self.voltages.shape[0] != batch_size:
            voltage_size = self.voltages.shape[0]
            repeats = int(batch_size / voltage_size)
            remainder = batch_size % voltage_size
            repetition_vector = torch.tensor([repeats], device=self.device).repeat(voltage_size)
            repetition_vector[-1] = repetition_vector[-1] + remainder

            self.voltages = torch.repeat_interleave(self.voltages, repetition_vector, dim=0).clone()
            self.errors = torch.repeat_interleave(self.errors, repetition_vector, dim=0).clone()
            self.noise = torch.repeat_interleave(self.noise, repetition_vector, dim=0).clone()
            self.Delta_rho = torch.repeat_interleave(self.rho_HP, repetition_vector, dim=0).clone()
            self.rho_HP = torch.repeat_interleave(self.rho_HP, repetition_vector, dim=0).clone()
            self.voltage_lookaheads = torch.repeat_interleave(self.voltage_lookaheads, repetition_vector, dim=0).clone()

    def _update_OU_noise(self, noise):
        """
        Calculate:
            layerwise noise:    xi[t] = xi[t-dt] + 1/tau_xi (\\sqrt(tau_xi dt) \\sigma_\\ell w - dt xi[t-dt])

        Args:
            noise: xi[t-dt]

        Returns:
            noise: xi[t]

        """

        # sample new noise and low-pass filter
        temp_noise = torch.empty(noise.size()).normal_(mean=0.0, std=1.0).to(self.device)
        noise += 1/self.tau_xi * (np.sqrt(self.tau_xi * self.dt) * self.sigma * temp_noise - self.dt * noise)
        # noise += 1/self.tau_xi * (np.sqrt(self.tau_xi * self.dt) * self.sigma * torch.randn(noise.size(), device=self.device) - self.dt * noise)

        return noise

    def _calc_HP(self, Delta_rho):
        """
        Calculate:
            high-pass of rho using tau_HP:
            rho_HP += Delta rho - dt / tau_HP rho_HP

        Args:
            rate Delta rho

        Returns:
            high-pass filtered rate: rho_HP

        """

        # sample new noise and low-pass filter
        # print(Delta_rho.size(), self.rho_HP.size())
        self.rho_HP += Delta_rho - self.dt/self.tau_HP * self.rho_HP

        return self.rho_HP

    def update_bw_weights(self, rho_HP, noise, with_optimizer=False):

        if self.train_B:

            dot_bw_weights = self.get_bw_weight_derivatives(rho_HP, noise)
            self.bw_weights.grad = -self.dt * dot_bw_weights  # our gradients are inverse to pytorch gradients

            if not with_optimizer:  # minus because pytorch gradients are inverse to our gradients
                self.bw_weights -= self.bw_weights.grad * self.learning_rate_B



    # ### CALCULATE WEIGHT DERIVATIVES ### #

    def get_weight_derivatives(self):
        """
        Return weight derivative calculated from current rate and errors.
        Args:

        Returns:
            weight_derivative: e * r^T * η

        """
        # If the input is served as a single sample, it is not in batch form, but this here requires it.
        if len(self.rho_input.shape) == 1:
            self.rho_input = self.rho_input.unsqueeze(0)
        return (torch.einsum('bi,bj->bij', self.rho_input, self.errors)).mean(0)

    # ### CALCULATE WEIGHT DERIVATIVES ### #

    def get_bw_weight_derivatives(self, rho_HP, noise):
        """
        Return weight derivative calculated from current rate and errors.
        Args:

        Returns:
            weight_derivative: noise * r_HP^T

        """

        return (torch.einsum('bi,bj->bij', rho_HP, noise)).mean(0) - self.regularizer * self.bw_weights

    def get_PAL_parameters(self):
        param_dict = {"learning_rate_bw": self.learning_rate_B,
                "regularizer": self.regularizer,
                "tau_xi": self.tau_xi,
                "tau_HP": self.tau_HP,
                "tau_LO": self.tau_LO,
                "sigma": self.sigma,
                "wn_sigma": self.wn_sigma}
        return param_dict

class LESequential(object):
    def __init__(self, layers, learning_rate, learning_rate_factors, loss_fn, loss_fn_deriv, tau, dt, beta, model_variant, target_type, with_optimizer=False, algorithm='BP', **kwargs):
        
        self.algorithm = algorithm
        self.tau = tau
        self.dt = dt
        self.beta = beta
        self.training_beta = beta

        if self.algorithm == 'PAL':
            self.bw_lr_factors = kwargs.get('bw_lr_factors', None)
            self.regularizer = kwargs.get('regularizer', None)
            self.tau_xi = [tau_xi * self.dt for tau_xi in kwargs.get('tau_xi', None)]     # PAL time constants are passed in multiples of dt
            self.tau_HP = [tau_HP * self.dt for tau_HP in kwargs.get('tau_HP', None)]
            self.tau_LO = [tau_LO * self.dt for tau_LO in kwargs.get('tau_LO', None)]
            self.sigma = kwargs.get('sigma', None)
        self.wn_sigma = kwargs.get('wn_sigma', None)

        self.layers = layers
        self.learning_rate_W = learning_rate
        self.learning_rate_biases = learning_rate
        for i, l in enumerate(layers):
            l.learning_rate_W = learning_rate * learning_rate_factors[i]
            l.learning_rate_biases = learning_rate * learning_rate_factors[i]
            l.tau = tau
            l.dt = dt
            if self.algorithm == 'PAL':
                l.learning_rate_B = learning_rate * self.bw_lr_factors[i]
                l.regularizer = self.regularizer[i]
                l.tau_xi = self.tau_xi[i]
                l.tau_HP = self.tau_HP[i]
                l.tau_LO = self.tau_LO[i]
                l.sigma = self.sigma[i]
                l.wn_sigma = self.wn_sigma[i]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_fn = loss_fn
        self.loss_fn_deriv = loss_fn_deriv
        self.layer_qty = len(self.layers)
        self.train_W = True
        if self.algorithm == 'PAL':
            self.train_B = True
        self.target = None

        layer_input_size = list(layers[0].input_size) if isinstance(layers[0].input_size, tuple) else [layers[0].input_size] # to handle projection, conv layer dimensions
        layer_input_size = [1] + layer_input_size  # batch dimension
        self.rho = [torch.zeros(layer_input_size, device=self.device)]
        if self.algorithm == 'PAL':
            self.rho_HP = [torch.zeros(layer_input_size, device=self.device)]
            self.noise = [torch.zeros(layer_input_size, device=self.device)]
        self.rho_deriv = [torch.zeros(layer_input_size, device=self.device)]
        self.errors = [torch.zeros(layer_input_size, device=self.device)]
        for l in self.layers:
            self.rho.append(torch.zeros([1, l.target_size], device=self.device))
            if self.algorithm == 'PAL':
                self.rho_HP.append(torch.zeros([1, l.target_size], device=self.device))
                self.noise.append(torch.zeros([1, l.target_size], device=self.device))
            self.rho_deriv.append(torch.zeros([1, l.target_size], device=self.device))
            self.errors.append(torch.zeros([1, l.target_size], device=self.device))

        self.dummy_label = torch.zeros([1, self.layers[-1].target_size], device=self.device)

        self.model_variant = model_variant
        self.target_type = target_type
        self.with_optimizer = with_optimizer

    @staticmethod
    def _set_random_seed(rnd_seed):
        """
        Set random seeds to frameworks.
        """
        np.random.seed(rnd_seed)
        torch.manual_seed(rnd_seed)

    def train(self):
        """
        Enables training.
        :return:
        """
        self.train_W = True
        if self.algorithm == 'PAL':
            self.train_B = True

        for l in self.layers:
            l.train()

    def eval(self):
        """
        Disables training.
        :return:
        """
        self.train_W = False
        if self.algorithm == 'PAL':
            self.train_B = False

        for l in self.layers:
            l.eval()

    def update(self, inputs, targets):
        """Performs an update step to the following equations defining the ODE of the neural network dynamics:

        :math:`(6.0) \\tau \\dot u = - u + W r +  e`

        :math:`(6.1)           e = r' \\odot  W^T[u - W r] + \\beta e^{trg}`

        :math:`(6.2)           e^{trg} = r' \\odot (r^{trg} - r)`

        Args:
            inputs: current input for layer
            targets: current target for layer

        Returns:

        """
        with torch.no_grad():  # thanks pytorch, we roll our own gradients
            if targets is None and self.train_W:
                raise Exception("For training, a valid target must be provided")
            elif not self.train_W:
                targets = self.dummy_label

            self.rho[0] = inputs.clone()
            self.rho_deriv[0] = torch.zeros_like(inputs)  # the first layer needs no rho deriv, because the calculated error can not be applied to any weight
            self._adapt_parallel_network_qty(inputs)

            target_error = self.beta * self._calculate_e_trg(targets, self.layers[-1].voltage_lookaheads, self.rho[-1], self.layers[-1].rho_deriv)

            if not self.train_W:
                target_error = target_error * 0.0

            if self.model_variant == ModelVariant.VANILLA:
                new_rhos = [None for _ in range(self.layer_qty + 1)]
                new_rho_derivs = [None for _ in range(self.layer_qty + 1)]
                if self.algorithm == 'PAL':
                    new_rho_HPs = [None for _ in range(self.layer_qty + 1)]
                    new_noises = [None for _ in range(self.layer_qty + 1)]

                for i, layer in enumerate(self.layers):
                    if self.algorithm == 'PAL':
                        new_rhos[i + 1], new_rho_derivs[i + 1], new_rho_HPs[i + 1], new_noises[i] = layer(self.rho[i], self.rho_deriv[i])
                    else:
                        new_rhos[i + 1], new_rho_derivs[i + 1] = layer(self.rho[i], self.rho_deriv[i])  # in principle W * r + b

                    if self.rho[i + 1] is None or new_rhos[i + 1].shape != self.rho[i + 1].shape:  # just so we get proper inits
                        self.rho[i + 1] = torch.zeros_like(new_rhos[i + 1])
                        self.rho_deriv[i + 1] = torch.zeros_like(new_rho_derivs[i + 1])
                        if self.algorithm == 'PAL':
                            self.rho_HP[i + 1] = torch.zeros_like(new_rho_HPs[i + 1])

                self.rho = new_rhos
                self.rho_deriv = new_rho_derivs
                if self.algorithm == 'PAL':
                    self.rho_HP = new_rho_HPs
                    self.noise = new_noises

            elif self.model_variant == ModelVariant.FULL_FORWARD_PASS:
                for i, layer in enumerate(self.layers):
                    self.rho[i + 1], self.rho_deriv[i + 1] = layer(self.rho[i], self.rho_deriv[i])  # in principle W * r + b
            else:
                raise Exception("Unknown model variant: one of vanilla or full_forward_pass")

            self._update_errors(target_error)

            if self.algorithm == 'PAL':
                self._update_bw_weights()


    def _update_errors(self, target_error):
        self.errors[-1] = target_error
        for i, layer in reversed(list(enumerate(self.layers))):
            self.errors[i] = layer.update_weights(self.errors[i + 1], self.with_optimizer)

    def _update_bw_weights(self):
        if self.algorithm == 'PAL':
            for i, layer in list(enumerate(self.layers))[1:]:
                layer.update_bw_weights(self.rho_HP[i + 1], self.noise[i - 1], self.with_optimizer)

    def get_errors(self):
        return self.errors

    def _calculate_e_trg(self, target, voltage_lookaheads, rho, rho_deriv):
        """
        Calculate e_trg (insert into errors as err[-self.target_size:] = get_e_trg(...).)
        Args:
            target:
            rho:
            rho_deriv:

        Returns:
            e_trg

        """

        # calculate target error
        # and use rate difference as target
        if self.target_type == TargetType.RATE:
            e_trg = rho_deriv * (target - rho)
        # or use softmax
        elif self.target_type == 'softmax':
            e_trg = target - F.softmax(voltage_lookaheads, dim=-1)
        # or use voltage difference as target
        elif self.target_type == 'mse':
            e_trg = (target - voltage_lookaheads)
        else:
            raise ValueError('target_type unknown, should be mse or softmax') 
        return e_trg

    def infer(self, x):
        self.eval()
        self.update(x, self.dummy_label)

    def __call__(self, x):
        return self.infer(x)

    # KERAS-like INTERFACE
    def fit(self, x=None, y=None, data_loader=None, n_updates: int = 100, batch_size=1, epochs=1, verbose=1):
        """
        Train network on dataset.

        Args:
            x: dataset of samples to train on.
            y: respective labels to train on.
            n_updates: number of weight updates per batch.
            batch_size: Number of examples per batch (batch training).
            epochs: Amount of epochs to train for.
            verbose: Level of verbosity.
        """
        if not self.train_W:  # if learning is totally off, then turn on learning with default values
            print("Learning off, turning on with beta {0}".format(self.beta))
            self.train()  # turn nudging on to enable training

        print("Learning with batch size {0}".format(batch_size))

        if data_loader is None and x is not None and y is not None:
            dataset = SimpleDataset(x, y)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=3, worker_init_fn=dataloader_seed_worker, drop_last=True)

        batch_qty = int(len(data_loader.dataset)/batch_size)
        for epoch_i in range(epochs):
            for batch_i, (x, y) in enumerate(data_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                self.fit_batch(x, y, n_updates, batch_i, batch_qty, verbose)

    def fit_batch(self, x=None, y=None, n_updates: int=100, batch_iteration=-1, batch_qty=-1, verbose: int=1):
        if verbose >= 1:
            print("train:: batch ", batch_iteration + 1 if batch_iteration != -1 else "", "/" if batch_qty != -1 else "", batch_qty if batch_qty != -1 else "", " | update ", end=" ")

        for update_i in range(n_updates):
            if verbose >= 2 and update_i % 10 == 0:
                print(update_i, end=" ")

            samples, labels = x, y
            self.update(samples, labels)

        if verbose >= 1:
            print('')

    def predict(self, x=None, data_loader=None, n_updates: int = 100, batch_size=1, verbose=1):
        """
        Predict batch with trained network.

        Args:
            x: samples to be predicted.
            n_updates: number of updates of the network used in tests.
            batch_size: Size of batch to be predicted.
            verbose: Level of verbosity.
        :return:
        """
        self.eval()   # turn nudging off to disable learning
        print("Learning turned off")

        if data_loader is None:
            n_samples = len(x)  # dataset size
            dataset = SimpleDataset(x, [np.zeros(self.layers[-1].target_size) for _ in range(n_samples)])
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=3, worker_init_fn=dataloader_seed_worker, drop_last=True)

        batch_qty = int(len(data_loader.dataset)/batch_size)
        predictions = []
        for batch_i, (x, _) in enumerate(data_loader):
            x = x.to(self.device)

            batch_predictions = self.predict_batch(x, batch_iteration=batch_i, batch_qty=batch_qty, n_updates=n_updates, verbose=verbose)
            predictions.extend(batch_predictions)

        return predictions

    def predict_batch(self, x, n_updates: int=100, batch_iteration=-1, batch_qty=-1, verbose: int=1):
        if verbose >= 1:
            print("predict:: batch ", batch_iteration + 1 if batch_iteration != -1 else "", "/" if batch_qty != -1 else "", batch_qty if batch_qty != -1 else "", " | update ", end=" ")

        for update_i in range(n_updates):
            if verbose >= 2 and update_i % 10 == 0:
                print(update_i, end=" ")

            samples = x
            self.update(samples, self.dummy_label)

        # use either rates or voltages as network output
        if self.target_type == TargetType.RATE:
            rates = self.rho[-1]
            batch_predictions = rates.detach().cpu().numpy()
        else:
            volts = self.layers[-1].voltage_lookaheads
            batch_predictions = volts.detach().cpu().numpy()

        if verbose >= 1:
            print('')

        return batch_predictions

    def _adapt_parallel_network_qty(self, x):
        """Adapt number of voltage sets to batch size (if more sets are required, repeat current sets, if less are required drop current sets).
        Returns:

        """
        batch_size = x.shape[0]
        if self.rho[1].shape[0] != batch_size:
            rho_size = self.rho[1].shape[0]
            repeats = int(batch_size / rho_size)
            remainder = batch_size % rho_size
            repetition_vector = torch.tensor([repeats], device=self.device).repeat(rho_size)
            repetition_vector[-1] = repetition_vector[-1] + remainder

            for i in range(1, len(self.rho)):
                self.rho[i] = torch.repeat_interleave(self.rho[i], repetition_vector, dim=0).clone()
                self.rho_deriv[i] = torch.repeat_interleave(self.rho_deriv[i], repetition_vector, dim=0).clone()
                if self.algorithm == 'PAL':
                    self.rho_HP[i] = torch.repeat_interleave(self.rho_HP[i], repetition_vector, dim=0).clone()
                    self.noise[i] = torch.repeat_interleave(self.noise[i], repetition_vector, dim=0).clone()
                self.errors[i] = torch.repeat_interleave(self.errors[i], repetition_vector, dim=0).clone()

            self.dummy_label = torch.repeat_interleave(self.dummy_label, repetition_vector, dim=0).clone()

    def save_model(self, logdir):
        for i, l in enumerate(self.layers):
            l.save_layer(logdir, i)
        now = datetime.now()
        current_time = str(now.strftime("%H:%M:%S"))
        print(f"saved at time: {str(current_time)}")

    def load_model(self, old_savedir):
        for (i, l) in enumerate(self.layers):
            l.load_layer(old_savedir, i)

    def parameters(self):
        params = []
        list(map(params.extend, [l.parameters() for l in self.layers]))
        return params
