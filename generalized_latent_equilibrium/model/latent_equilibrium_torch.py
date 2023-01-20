#!/usr/bin/env python
# PyTorch implementation of the Latent Equilibrium model. Trainable only with single samples.

# Authors: Benjamin Ellenberger (benelot@github) & Paul Haider (paulhaider@github)

import time

import numpy as np
import torch
import torch.nn as nn

from model.network_params import NetworkParams, ActivationFunction, ArchType, ModelVariant, TargetType

from torch.utils.data import DataLoader
from utils.torch_bottleneck import bottleneck
from utils.torch_utils import get_torch_dtype, SimpleDataset


class LatentEqNetwork:
    """
    Main class for network model simulations. Implements the ODEs of the Latent Equilibrium model.
    Prefixes: st = state, pl = placeholder, arg = argument.
    """

    # INIT
    ########
    def __init__(self, params: NetworkParams):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.params = params

        # store important params
        self.layers = params.layers
        self.tau = params.arg_tau  # ms
        self.dt = params.arg_dt  # ms
        self.set_beta(params.arg_beta)
        self.training_beta = params.arg_beta

        # learning rates
        self.arg_lrate_W = params.arg_lrate_W
        self.arg_lrate_B = params.arg_lrate_B
        self.arg_lrate_biases = params.arg_lrate_biases

        self.arg_w_init_params = params.arg_w_init_params
        self.arg_b_init_params = params.arg_b_init_params

        self.dtype = get_torch_dtype(params.dtype)

        self.use_biases = params.use_biases

        self.target_type = params.target_type
        self.model_variant = params.model_variant

        # setup integration method and activation function
        self.integration_method = params.integration_method
        self.activation_function = params.activation_function  # type of the activation function
        self.act_function, self.act_func_deriv, self.act_func_second_deriv = self._generate_activation_function(params.activation_function)

        self.network_architecture = params.network_architecture

        self._set_random_seed(params.rnd_seed)

        # setup inputs and outputs
        self.input_size = params.layers[0]  # input neurons
        self.target_size = params.layers[-1]  # target neurons
        self.neuron_qty = sum(params.layers)
        self.st_input = torch.zeros(self.input_size, dtype=self.dtype, device=self.device)
        self.st_old_input = torch.zeros(self.input_size, dtype=self.dtype, device=self.device)
        self.st_target = torch.zeros(self.target_size, dtype=self.dtype, device=self.device)
        self.st_old_target = torch.zeros(self.target_size, dtype=self.dtype, device=self.device)

        assert len(params.layers) - 1 == len(params.learning_rate_factors)

        # prepare input vector to generate input transients
        self.xr = torch.linspace(-20, 90, 100, dtype=self.dtype, device=self.device)
        self.old_x = torch.zeros(self.input_size, dtype=self.dtype, device=self.device)
        self.old_y = torch.zeros(self.target_size, dtype=self.dtype, device=self.device)
        self.dummy_label = torch.zeros(self.target_size, dtype=self.dtype, device=self.device)

        # setup neuron weight and bias masks
        self.weight_mask = self._make_connection_weight_mask(params.layers, params.learning_rate_factors, params.network_architecture, params.only_discriminative);  """mask for weights (matrix)"""
        self.bias_mask = self._make_bias_mask(params.layers, self.input_size, params.learning_rate_factors, params.only_discriminative);  """mask for biases (vector)"""

        # dictionary for logging
        self.logs = {}

        # setup network
        self._initialize_network()

        # perform basic profiling
        if params.check_with_profiler:
            print("\nProfile model for single sample:\n--------------------------------")
            self._profile_model()

        # perform single sample test run to get network performance stats
        self.time_per_prediction_step, self.time_per_train_step = self._test_simulation_run()
        self.report_simulation_params()

        # reinitialize network variables
        self._initialize_network()

    # NETWORK SETUP METHODS
    ###########################

    def _make_connection_weight_mask(self, layers, learning_rate_factors, network_architecture, only_discriminative=True):
        """
        Create weight mask encoding of the network structure.
        Weights are given as W[i][j], i = postsyn. neuron, j = presyn. neuron.
        mask[i][j] > 0 if a connection from neuron j to i exists, otherwise = 0.
        """
        if network_architecture == ArchType.LAYERED_FEEDFORWARD:                       # proper feed forward mask
            weight_mask = self._make_feed_forward_mask(layers, learning_rate_factors)  # make a feed forward mask

        elif network_architecture == ArchType.LAYERED_FEEDBACKWARD:
            weight_mask = self._make_feed_forward_mask(layers, learning_rate_factors)
            weight_mask = weight_mask.T

        elif network_architecture == ArchType.LAYERED_RECURRENT:                       # Recurrence within each layer mask
            weight_mask = self._make_feed_forward_mask(layers, learning_rate_factors)  # make a feed forward mask
            weight_mask += weight_mask.T                                               # make recurrent connections among the neurons of one layer

        elif network_architecture == ArchType.LAYERED_RECURRENT_RECURRENT:
            weight_mask = self._make_feed_forward_mask(layers, learning_rate_factors)
            weight_mask += weight_mask.T
            weight_mask = self._add_recurrent_connections(weight_mask, layers, learning_rate_factors)

        elif network_architecture == ArchType.IN_RECURRENT_OUT:
            weight_mask = self._make_feed_forward_mask(layers, learning_rate_factors)
            weight_mask = self._add_recurrent_connections(weight_mask, layers, learning_rate_factors)

        elif network_architecture == ArchType.FULLY_RECURRENT:                         # All to ALL connection mask without loops
            weight_mask = np.ones((sum(layers), sum(layers)))                          # create a complete graph mask of weights
            np.fill_diagonal(weight_mask, 0)                                           # drop loops

        else:
            raise NotImplementedError("Mask type ", network_architecture.name, " not implemented.")

        # only discriminative = no connections projecting back to the visible layer.
        if only_discriminative:
            weight_mask[:layers[0], :] *= 0

        weight_mask = torch.tensor(weight_mask, dtype=self.dtype, device=self.device)
        return weight_mask

    @staticmethod
    def _make_feed_forward_mask(layers, learning_rate_factors):
        """
        Returns a mask for a feedforward architecture.

        Args:
            layers: is a list containing the number of neurons per layer.
            learning_rate_factors: contains learning rate multipliers for each layer.

        Adapted from Jonathan Binas (@MILA)

        Weight mask encoding of the network structure
        ---------------------------------------------
        Weights are given as W[i][j], i = postsyn. neuron, j = presyn. neuron.
        mask[i][j] > 0 if a connection from neuron j to i exists, otherwise = 0.
        """
        neuron_qty = int(np.sum(layers))  # total quantity of neurons
        layer_qty = len(layers)  # total quantity of layers
        mask = np.zeros((neuron_qty, neuron_qty))  # create adjacency matrix of neuron connections
        layer_offsets = [0] + [int(np.sum(layers[:i + 1])) for i in range(layer_qty)]  # calculate start of layers, the postsynaptic neurons to each layer
        for i in range(len(learning_rate_factors)):
            mask[layer_offsets[i + 1]:layer_offsets[i + 2], layer_offsets[i]:layer_offsets[i + 1]] = learning_rate_factors[i]  # connect all of layer i (i to i+1) with layer i + 1

        return mask

    @staticmethod
    def _add_recurrent_connections(weight_mask, layers, learning_rate_factors):
        """
        Adds recurrent structure to every layer of network.
        """
        layer_qty = len(layers)  # total quantity of layers
        layer_offsets = [0] + [int(np.sum(layers[:i + 1])) for i in range(layer_qty)]  # calculate start of layer offsets

        for i in range(1, len(learning_rate_factors)):  # exclude input layer
            weight_mask[layer_offsets[i]:layer_offsets[i + 1], layer_offsets[i]:layer_offsets[i + 1]] = learning_rate_factors[i]  # connect all of layer i with itself
        np.fill_diagonal(weight_mask, 0)  # drop loops (autosynapses)
        return weight_mask

    def _make_bias_mask(self, layers, input_size, learning_rate_factors, only_discriminative):
        """
        Creates mask for biases (similar to mask for weights).
        """
        neuron_qty = int(np.sum(layers))  # total quantity of neurons
        layer_qty = len(layers)  # total quantity of layers
        bias_mask = np.ones(neuron_qty)
        layer_offsets = [0] + [int(np.sum(layers[:i + 1])) for i in range(layer_qty)]  # calculate start of layers
        for i in range(len(learning_rate_factors)):
            bias_mask[layer_offsets[i]:layer_offsets[i + 1]] *= learning_rate_factors[i]  # set learning rate factors for biases of each layer

        if only_discriminative:
            bias_mask[:input_size] *= 0

        bias_mask = torch.tensor(bias_mask, dtype=self.dtype, device=self.device)
        return bias_mask

    # # ODE PARTS # #
    # ## ACTIVATION FUNCTIONS AND DERIVATIVES ##
    def _generate_activation_function(self, activation_function):
        """
        Implementation of different activation functions.
        """
        if activation_function == ActivationFunction.SIGMOID:  # if the activation is a sigmoid
             act_function = nn.Sigmoid()  # define the activation function as a sigmoid of voltages
             act_func_deriv = lambda voltages: act_function(voltages) * (1 - act_function(voltages))  # function of the 1st derivative
             act_func_second_deriv = lambda voltages: act_func_deriv(voltages) * (1 - 2 * act_function(voltages))  # function of the 2nd derivative
        elif activation_function == ActivationFunction.RELU:  # regular ReLu unit
            act_function = nn.ReLU()
            act_func_deriv = lambda voltages: torch.tensor((voltages > 0) * 1, dtype=self.dtype, device=self.device)  # compute the derivative of the relu function
            act_func_second_deriv = lambda voltages: 0.0
        elif activation_function == ActivationFunction.HARD_SIGMOID:  # ReLU which is clipped to 0-1
            act_function = lambda voltages: voltages.clamp(0, 1)
            act_func_deriv = lambda voltages: torch.tensor((voltages >= 0) * (voltages <= 1), dtype=self.dtype, device=self.device)
            act_func_second_deriv = lambda voltages: voltages * 0.0
        else:
            raise ValueError('The activation function type _' + activation_function.name + '_ is not implemented!')

        return act_function, act_func_deriv, act_func_second_deriv

    def _initialize_network(self):
        """
        Set up voltages, weights, and their derivatives.
        """
        # states
        # setup voltage variable for neurons
        self.st_voltages = torch.zeros(self.neuron_qty, dtype=self.dtype, device=self.device)  # initialize all voltages to the resting mb potential, 0
        self.st_voltages_deriv = torch.zeros(self.neuron_qty, dtype=self.dtype, device=self.device)  # initialize all voltage derivatives to 0
        self.st_voltage_lookaheads = torch.zeros(self.neuron_qty, dtype=self.dtype, device=self.device)  # initialize all voltage derivatives to 0
        self.st_synaptic_current = torch.zeros(self.neuron_qty, dtype=self.dtype, device=self.device)
        self.st_basal_inputs = torch.zeros(self.neuron_qty, dtype=self.dtype, device=self.device)
        self.st_weights = self._create_initial_weights(self.weight_mask,
                                                       self.arg_w_init_params['mean'],
                                                       self.arg_w_init_params['std'],
                                                       self.arg_w_init_params['clip'])
        self.st_weights = torch.tensor(self.st_weights, dtype=self.dtype, device=self.device)
        # setup bias variables for neurons
        self.st_biases = self._create_initial_biases(self.use_biases, self.bias_mask,
                                                     self.arg_b_init_params['mean'],
                                                     self.arg_b_init_params['std'],
                                                     self.arg_b_init_params['clip'])
        self.st_biases = torch.tensor(self.st_biases, dtype=self.dtype, device=self.device)

        self.rho = torch.zeros(self.neuron_qty, dtype=self.dtype, device=self.device)                         # rates
        self.rho_deriv = torch.zeros(self.neuron_qty, dtype=self.dtype, device=self.device)                   # rate derivatives
        self.errors = torch.zeros(self.neuron_qty, dtype=self.dtype, device=self.device)                      # errors

    @staticmethod
    def _create_initial_weights(weight_mask, mean, std, clip):
        """
        Create randomly initialized weight matrix.
        """
        neuron_qty = weight_mask.shape[0]
        return np.clip(np.random.normal(mean, std, size=(neuron_qty, neuron_qty)), -clip, clip) * (weight_mask.cpu().numpy() > 0)  # initialize weights with normal sample where mask is larger 0

    @staticmethod
    def _create_initial_biases(use_biases, bias_mask, mean, std, clip):
        """
        Create randomly initialized bias matrix (or set biases to zero if only weights are used).
        """
        neuron_qty = bias_mask.shape[0]
        if use_biases:
            return np.clip(np.random.normal(mean, std, size=neuron_qty), -clip, clip) * (bias_mask.cpu().numpy() > 0)  # initialize biases with normal sample where mask is larger 0
        else:
            return np.zeros(neuron_qty)  # set biases to zero

    # BASIC PROFILE AND TEST METHODS
    ############################################
    def _profile_model(self):
        """
        Profiling for the comp. graph. Performs one network update step and saves the profile data.
        The data can be loaded in Chrome at chrome://tracing/.
        """
        sample_input = np.ones(self.input_size)  # input voltages set to 1
        sample_output = np.ones(self.target_size)  # output voltages set to 1

        _, cpu_profiler, cuda_profiler = bottleneck(lambda: self.update_network(sample_input, sample_output, True))

        if cpu_profiler is not None:
            cpu_profiler.export_chrome_trace('torch_cpu_timeline.json')

        if cuda_profiler is not None:
            cuda_profiler.export_chrome_trace('torch_cuda_timeline.json')

    def _test_simulation_run(self):
        """
        Test network run. Estimates the average time used to perform a time/integration step.
        """
        sample_size = 50.0  # number of samples to calculate the average integration time of

        sample_input = np.ones(self.input_size)  # input voltages set to 1
        sample_output = np.ones(self.target_size)  # output voltages set to 1

        # test prediction
        init_time = time.time()  # initial time
        for _ in range(int(sample_size)):
            self.update_network(sample_input, sample_output, False)  # run sample_size prediction updates of the network

        time_per_prediction_step = (time.time() - init_time) / sample_size  # compute average time for one iteration

        # test training
        init_time = time.time()
        for _ in range(int(sample_size)):
            self.update_network(sample_input, sample_output, True)  # run sample_size training updates of the network

        time_per_train_step = (time.time() - init_time) / sample_size  # compute average time for one iteration

        return time_per_prediction_step, time_per_train_step

    def report_simulation_params(self):
        """
        Print report of simulation setup to the terminal.
        """
        print('------------------')
        print('NETWORK PARAMETERS')
        print('------------------')
        print('Total number neurons: ', self.neuron_qty)
        print('Total number syn. connections: ', int(torch.sum(self.weight_mask).item()))
        print('Layer structure: ', self.layers)
        print('Network architecture: ', self.network_architecture.name)
        print('Model variant: ', self.model_variant.name)
        print('Target type: ', self.target_type.name)
        print('Integration method: ', self.integration_method.name)
        print('Activation function: ', self.activation_function.name)
        print('Weight initial distribution: ', self.arg_w_init_params)
        print('Use biases: ', self.use_biases)
        print('Bias initial distribution: ', self.arg_b_init_params)
        print('Learning rate: ', self.arg_lrate_W)
        print('Beta (nudging parameter): ', self.beta)
        print('Rate time constant (Tau): {0} ms'.format(self.tau))
        print('Time step: {0} ms'.format(self.dt))
        print('Time per prediction step in test run: {0} s'.format(self.time_per_prediction_step))
        print('Time per training step in test run: {0} s'.format(self.time_per_train_step))
        print('------------------')
        print("Simulation framework: Torch ", torch.__version__)
        print('Simulation running on :', self.device)
        print('------------------')

    # PERFORM UPDATE STEP OF NETWORK DYNAMICS
    #########################################
    def _perform_update_step(self, train_W=False):
        """Performs an update step to the following equations defining the ODE of the neural network dynamics:

        :math:`(6.0) \\tau \\dot u = - u + W r +  e`

        :math:`(6.1)           e = r' \\odot  W^T[u - W r] + \\beta e^{trg}`

        :math:`(6.2)           e^{trg} = r' \\odot (r^{trg} - r)`

        Args:
            train_W:

        Returns:

        """

        self.st_synaptic_current = torch.tensor(self.rho, dtype=self.dtype, device=self.device)

        # CALCULATE VOLTAGE DERIVATIVES AND UPDATE VOLTAGES
        # TODO[BE]: Initializations were needed for numba to know the size, here we do not need it?

        voltages = torch.zeros(self.neuron_qty, dtype=self.dtype)
        basal_inputs = torch.zeros(self.neuron_qty, dtype=self.dtype)
        lookahead_voltages = torch.zeros(self.neuron_qty, dtype=self.dtype)
        voltages_deriv = torch.zeros(self.neuron_qty, dtype=self.dtype)
        layerwise_syn_current = torch.zeros(self.neuron_qty, dtype=self.dtype)

        # select chosen model variant
        if self.model_variant == ModelVariant.VANILLA:
            # vanilla implementation of LE
            basal_inputs = self._calculate_basal_inputs(self.st_synaptic_current, self.st_weights, self.st_biases)  # W * r + b
            voltages_deriv = 1.0 / self.tau * (basal_inputs - self.st_voltages + self.errors)
            lookahead_voltages = self.st_voltages + self.tau * voltages_deriv  # U_t = u_{t-1} + τ dot{u}_{t-1}
            voltages = self.st_voltages + self.dt * voltages_deriv  # u_t = u_{t-1} + dt dot{u}_{t-1}
        elif self.model_variant == ModelVariant.FULL_FORWARD_PASS:
            # calculate forward pathway similar to deep learning
            # and add errors in between
            layerwise_inputs = torch.zeros(self.neuron_qty, dtype=self.dtype, device=self.device)
            for layer in range(len(self.layers)):
                # calculate layerwise input
                if layer == 0:
                    layerwise_inputs[:self.input_size] = self.st_input  # external input
                else:
                    layerwise_inputs = torch.mv(self.st_weights, layerwise_syn_current)  # internal input (W_i * r) + b
                    layerwise_inputs += (layerwise_inputs != 0) * self.st_biases  # add biases

                # perform Euler step for each layer to integrate somatic voltage
                layerwise_volts = (layerwise_inputs != 0) * self.st_voltages
                layerwise_errors = (layerwise_inputs != 0) * self.errors
                layerwise_deriv = 1.0 / self.tau * (layerwise_inputs - layerwise_volts + layerwise_errors)
                layerwise_volts += self.dt * layerwise_deriv  # u_t = u_{t-1} + dt dot{u}_{t-1}
                layerwise_lookahead = layerwise_volts + self.tau * layerwise_deriv
                layerwise_rate = (layerwise_inputs != 0) * self.act_function(layerwise_lookahead)
                layerwise_syn_current = layerwise_rate.copy()

                # store layerwise results in full vector
                basal_inputs += layerwise_inputs
                voltages += layerwise_volts
                lookahead_voltages += layerwise_lookahead
        else:
            raise NotImplementedError(f"Model variant {self.model_variant} not implemented.")

        # update state voltages and rates
        self.st_voltages = voltages
        self.st_voltages_deriv = voltages_deriv
        self.st_basal_inputs = basal_inputs
        self.st_voltage_lookaheads = lookahead_voltages

        # calculate rates
        self.rho = self.act_function(self.st_voltage_lookaheads)  # ρ(U)
        self.rho_deriv = self.act_func_deriv(self.st_voltage_lookaheads)  # ρ'(U)

        # CALCULATE ERRORS
        errors = torch.zeros(self.neuron_qty, dtype=self.dtype)
        errors = self._get_errors()

        # update errors
        self.errors = errors

        # CALCULATE WEIGHT DERIVATIVES AND UPDATE WEIGHTS
        if train_W:
            dot_weights = self.get_weight_derivatives()
            new_weights = self.st_weights + self.dt * dot_weights

        else:
            dot_weights = torch.zeros((self.neuron_qty, self.neuron_qty), dtype=self.dtype, device=self.device)
            new_weights = self.st_weights

        # update weights
        self.st_weights = new_weights

        # CALCULATE BIAS DERIVATIVE AND UPDATE BIASES
        if self.use_biases:  # if we use biases
            if train_W:
                dot_biases = self.get_bias_derivatives()  # do bias update
                new_biases = self.st_biases + self.dt * dot_biases
            else:
                # dot_biases = np.zeros(self.neuron_qty)
                new_biases = self.st_biases
            self.st_biases = new_biases

        # print(f"Volt deriv: {voltages_deriv[-self.target_size:].detach().cpu().numpy()}")
        # print(f"Error deriv: {self.errors[-self.target_size:].detach().cpu().numpy()}")
        # print(f"Target-lookahead diff: {(self.st_target[-self.target_size:] - self.st_voltage_lookaheads[-self.target_size:]).detach().cpu().numpy()}\n")

        return voltages, voltages_deriv, new_weights, dot_weights

    # FUNCTIONS OF CALCULATIONS NEEDED TO SOLVE ODE
    ########################################################

    def set_input_and_target(self, input, target):
        """
        Set input and target of the network.
        Note: We need the input of the previous time step to approximate the derivatives of the inputs.
        Args:
            input:
            target:

        Returns:

        """
        # assert that input and output voltages are tensors
        if isinstance(input, np.ndarray):
            input = torch.tensor(input, dtype=self.dtype, device=self.device)

        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=self.dtype, device=self.device)

        # set (formerly) current as old and new input as current
        self.st_old_input = self.st_input
        self.st_input = input

        self.st_old_target = self.st_target
        self.st_target = target

    # ### CALCULATE WEIGHT DERIVATIVES ### #

    def get_weight_derivatives(self):
        """
        Return weight derivative calculated from current rate and errors.
        Args:

        Returns:
            weight_derivative: e * r^T * η * weight_mask

        """
        return torch.outer(self.errors, self.rho) * self.weight_mask * self.arg_lrate_W

    def _calculate_weight_derivatives(self, voltage_lookaheads, weights, biases):
        """
        Calculate weight derivative (Eq. 18).
        Args:
            voltage_lookaheads: U
            weights: W
            biases: b

        Returns:
            weight_derivative: (U - (W * r + b)) * r^T * η * weight_mask

        """
        rho = self.act_function(voltage_lookaheads)  # r = ρ(U)
        basal_inputs = self._calculate_basal_inputs(rho, weights, biases)  # W * r
        return torch.outer(voltage_lookaheads - basal_inputs, rho) * self.weight_mask * self.arg_lrate_W

    def _calculate_basal_inputs(self, rho, weights, biases):
        """
        Calculate the inputs coming from other layers to each layer (W_i * r) + u_input.
        Args:
            rho:
            weights:

        Returns:

        """
        basal_input = torch.mv(weights, rho) + biases  # internal input (W_i * r) + b
        # here we clamp the basal input to be the input rate (as opposed to internal input + input rate)
        # therefore the input neurons receive only defined input and none from the network
        basal_input[:self.input_size] = self.st_input  # internal input + external input = total input
        return basal_input

    # ### CALCULATE BIAS DERIVATIVES ### #

    def get_bias_derivatives(self):
        """
        Return bias derivative.
        Args:

        Returns:
            bias_derivative: e * η * bias_mask

        """
        return self.errors * self.bias_mask * self.arg_lrate_biases

    def _calculate_bias_derivatives(self, voltage_lookaheads, weights, biases):
        """
        Calculate bias derivative.
        Same as weights_update but without pre-synaptic activities (Eq. 18)
        Args:
            voltage_lookaheads: U
            weights: W
            biases: b

        Returns:
            bias_derivative: (U - (W * r + b)) * η * weight_mask

        """
        rho = self.act_function(voltage_lookaheads)  # ρ(U)
        basal_inputs = self._calculate_basal_inputs(rho, weights, biases)  # W * r
        return (voltage_lookaheads - basal_inputs) * self.bias_mask * self.arg_lrate_biases

    # SOLVE ODE WITHOUT SOLVER
    ###########################

    def _get_lookahead_voltages(self, voltages, dot_voltages):
        """
        Calculate voltages lookaheads.
        Get u = \\bar u + \\tau \\dot{\\bar u}
        Args:
            voltages:
            dot_voltages:

        Returns:

        """
        return voltages + self.tau * dot_voltages

    def _get_errors(self):
        """
        Calculate network error:
            layerwise error:    e = diag(r') W^T (U - Wr) + β e^{trg}

        Args:

        Returns:
            errors:

        """
        errors = self._calculate_errors(self.st_voltages, self.st_voltage_lookaheads, self.rho, self.rho_deriv, self.st_target, self.st_basal_inputs)
        return errors

    def _calculate_errors(self, voltages, voltage_lookaheads, rho, rho_deriv, target, basal_inputs):
        """
        Calculate:
            layerwise error:    e = diag(r') W^T (U - Wr) + β e^{trg}

            target error:       e^{trg} = diag(r') (r^{trg} - r) using target rate or
                                e^{trg} = diag(r') (U^{trg} - U) using target voltage

        Args:
            voltage_lookaheads:
            rho:
            rho_deriv:
            target:
            basal_inputs:

        Returns:
            errors:

        """
        # e (missing e_{trg})
        err = rho_deriv * torch.mv(self.st_weights.t(), voltage_lookaheads - basal_inputs)

        # + e_{trg}
        err[-self.target_size:] = self.beta * self._calculate_e_trg(target, voltages, voltage_lookaheads, rho, rho_deriv)
        return err

    # INDIVIDUAL PARTS OF THE ODE
    ###############################

    def _get_e_trg(self):
        """
        Return target error
        Args:

        Returns:
            e_trg

        """
        e_trg = self._calculate_e_trg(self.st_target, self.st_voltages, self.st_voltage_lookaheads, self.rho, self.rho_deriv)
        return e_trg

    def _calculate_e_trg(self, target, voltages, voltage_lookaheads, rho, rho_deriv):
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
            e_trg = rho_deriv[-self.target_size:] * (target - rho[-self.target_size:])
        # or use voltage difference as target
        else:
            e_trg = (target - voltage_lookaheads[-self.target_size:])
        return e_trg

    # NETWORK INTEGRATOR
    #####################

    # use target rate instead of target voltage in LE framework
    def update_network(self, input_current, target_rate, train_W=False):
        """
        Perform a single integration step.
        """
        self.set_input_and_target(input_current, target_rate)
        self._perform_update_step(train_W)

    @staticmethod
    def _set_random_seed(rnd_seed):
        """
        Set random seeds to frameworks.
        """
        np.random.seed(rnd_seed)
        torch.manual_seed(rnd_seed)

    # GET and SET
    def set_beta(self, beta):
        """
        Set the nudging beta.
        Args:
            beta:

        Returns:

        """
        self.beta = beta

    def get_beta(self):
        return self.beta

    def set_lrate_W(self, learning_rate):
        self.arg_lrate_W = learning_rate

    def set_lrate_biases(self, learning_rate):
        self.arg_lrate_biases = learning_rate

    def get_lrate_W(self):
        return self.arg_lrate_W

    def get_lrate_biases(self):
        return self.arg_lrate_biases

    def get_voltages(self):
        return self.st_voltages.detach().cpu().numpy()

    def set_voltages(self, voltages):
        if self.st_voltages.shape != voltages.shape:
            raise Exception(f'Incompatible voltage shapes {self.st_voltages.shape} {voltages.shape}')

        if isinstance(voltages, np.ndarray):
            voltages = torch.tensor(voltages, dtype=self.dtype, device=self.device)

        self.st_voltages = voltages

    def get_voltage_lookaheads(self):
        return self.st_voltage_lookaheads.detach().cpu().numpy()

    def get_target(self):
        return self.st_target.detach().cpu().numpy()

    def get_output_voltages(self):
        """
        Get the voltages of the output neurons.

        :math:`u(t)_{output}`

        Returns:
            output neuron voltages
        """
        output_volts = self.st_voltage_lookaheads[-self.layers[-1]:]

        return output_volts.detach().cpu().numpy()

    def get_rates(self):
        return self.act_function(self.st_voltage_lookaheads)

    def get_output_rates(self):
        return self.act_function(self.get_output_voltages())

    def set_weights(self, weights):
        if self.st_weights.shape != weights.shape:
            raise Exception(f'Incompatible weights shapes {self.st_weights.shape} {weights.shape}')

        if isinstance(weights, np.ndarray):
            weights = torch.tensor(weights, dtype=self.dtype, device=self.device)

        self.st_weights = weights

    def get_weights(self):
        return self.st_weights.detach().cpu().numpy()

    def get_biases(self):
        return self.st_biases.detach().cpu().numpy()

    def set_biases(self, biases):
        if self.st_biases.shape != biases.shape:
            raise Exception(f'Incompatible biases shapes {self.st_biases.shape} {biases.shape}')

        if isinstance(biases, np.ndarray):
            biases = torch.tensor(biases, dtype=self.dtype, device=self.device)

        self.st_biases = biases

    def get_voltage_derivatives(self):
        self.st_voltages_deriv.detach().cpu().numpy()

    def get_errors(self):
        self.errors.detach().cpu().numpy()

    def get_dot_voltages(self):
        self.get_voltage_derivatives()

    def get_output(self):
        """
        Returns network output based on target definition.

        Returns:
            output

        """
        # choose output based on target type
        if self.target_type == TargetType.RATE:
            # return output rate
            output = self.get_output_rates()
        else:
            # or output voltage
            output = self.get_output_voltages()

        return output

    # KERAS-like INTERFACE
    def fit(self, x=None, y=None, n_updates: int = 100, batch_size=1, epochs=1, verbose=1, is_timeseries=False):
        """
        Train network on dataset.

        Args:
            x: dataset of samples to train on.
            y: respective labels to train on.
            n_updates: number of weight updates per sample or rather sample presentation time.
            epochs: Amount of epochs to train for.
            verbose: Level of verbosity.
            is_timeseries: if the input is a time series
        """
        n_samples = len(x)  # dataset size

        if self.beta == 0:  # if learning is totally off, then turn on learning with default values
            print("Learning off, turning on with beta {0}".format(self.training_beta))
            self.set_beta(self.training_beta)  # turn nudging on to enable training

        print("Learning with single samples")

        dataset = SimpleDataset(x, y)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=3)

        for epoch_i in range(epochs):
            for sample_i, (x, y) in enumerate(data_loader):
                x = x[0].to(self.device)  # put first and only sample to device
                y = y[0].to(self.device)
                if sample_i == 0:
                    self.old_x = x
                    self.old_y = y

                if verbose >= 1:
                    print("train:: sample ", sample_i + 1, "/", n_samples, " | update ", end=" ")

                if self.xr.shape[0] != n_updates:  # update transients input vector to number of updates
                    self.xr = torch.linspace(-20, 90, n_updates, dtype=self.dtype, device=self.device)

                for update_i in range(n_updates):
                    if verbose >= 2 and update_i % 10 == 0:
                        print(update_i, end=" ")

                    if is_timeseries:
                        sample, label = x, y
                    else:
                        sample, label = self._decay_func(self.xr[update_i], self.old_x, x), self._decay_func(self.xr[update_i], self.old_y, y)
                    self.update_network(sample, label, train_W=True)

                self.old_x = x
                self.old_y = y

                if verbose >= 1:
                    print('')

    def fit_batch(self, x=None, y=None, n_updates: int=100, batch_iteration=-1, batch_qty=-1, verbose: int=1, is_timeseries: bool=False):
        raise NotImplementedError

    def predict(self, x=None, n_updates: int = 100, batch_size=1, verbose=1, is_timeseries=False):
        """
        Predict batch with trained network.

        Args:
            x: samples to be predicted.
            n_updates: number of updates of the network used in tests.
            verbose: Level of verbosity.
            is_timeseries: if the input is a time series
        :return:
        """
        n_samples = len(x)  # dataset size
        self.set_beta(0.0)   # turn nudging off to disable learning
        print("Learning turned off")

        dataset = SimpleDataset(x, [np.zeros(self.target_size) for i in range(n_samples)])
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=3)

        predictions = []
        for sample_i, (x, _) in enumerate(data_loader):
            x = x[0].to(self.device)  # put first and only sample to device
            if sample_i == 0:
                self.old_x = x

            if verbose >= 1:
                print("predict:: sample", sample_i + 1, "/", n_samples, " | update ", end=" ")

            if self.xr.shape[0] != n_updates:  # update transients input vector to number of updates
                self.xr = torch.linspace(-20, 90, n_updates, dtype=self.dtype, device=self.device)

            for update_i in range(n_updates):
                if verbose >= 2 and update_i % 10 == 0:
                    print(update_i, end=" ")

                if is_timeseries:
                    sample = x
                else:
                    sample = self._decay_func(self.xr[update_i], self.old_x, x)
                self.update_network(sample, self.dummy_label, train_W=False)

            self.old_x = x

            # use either rates or voltages as network output
            if self.target_type == TargetType.RATE:
                rates = self.get_rates()
                prediction = rates[-self.target_size:].detach().cpu().numpy()
            else:
                volts = self.st_voltage_lookaheads
                prediction = volts[-self.target_size:].detach().cpu().numpy()

            predictions.append(prediction)

            if verbose >= 1:
                print('')

        return predictions

    def predict_batch(self, x, n_updates: int = 100, batch_iteration=-1, batch_qty=-1, verbose: int=1, is_timeseries: bool=False):
        raise NotImplementedError

    def get_mse_loss(self):
        """
        Return mean-squared-error (MSE) loss.
        Args:

        Returns:
            mse_loss

        """
        loss = torch.sum(torch.pow((self.st_target - self.get_output()), 2))
        return loss

    def __call__(self, x, n_updates: int = 100, verbose=1, is_timeseries=False):
        self.predict(x, n_updates, verbose, is_timeseries)

    # SAVE AND LOAD NETWORK
    def save(self, save_path):
        """
        Save the lagrange model to file.
        Args:
            save_path:

        Returns:

        """
        voltages = self.st_voltages.detach().cpu().numpy()
        weights = self.st_weights.detach().cpu().numpy()
        biases = self.st_biases.detach().cpu().numpy()

        np.save('{0}/voltages'.format(save_path), voltages)
        np.save('{0}/weights'.format(save_path), weights)
        np.save('{0}/biases'.format(save_path), biases)

    def load(self, load_path):
        """
        Load the lagrange model from file.
        Args:
            load_path:

        Returns:

        """
        try:
            voltages = np.load('{0}/voltages.npy'.format(load_path))
            weights = np.load('{0}/weights.npy'.format(load_path))
            biases = np.load('{0}/biases.npy'.format(load_path))
        except Exception:
            return False

        self.st_voltages = torch.tensor(voltages, device=self.device)
        self.st_weights = torch.tensor(weights, device=self.device)
        self.st_biases = torch.tensor(biases, device=self.device)

        return True

    def deepcopy(self):
        """
        Generate a deep copy of the network.

        Returns:
            the deepcopy of the network
        """
        n = LatentEqNetwork(self.params)
        n.set_weights(self.get_weights())
        n.set_biases(self.get_biases())
        n.set_voltages(self.get_voltages())

        return n

    # TIME SERIES BUILDER FUNCTIONS
    @staticmethod
    def _decay_func(x, equi1, equi2):
        """
        Decay function to transform sample and label by exponential fading.
        Sample x from xr2 = np.linspace(-20, 90, n_updates)
        """
        return equi1 + (equi2 - equi1) / (1 + torch.exp(-x / 4.0))
