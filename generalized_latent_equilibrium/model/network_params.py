from enum import Enum
import json
import re
import numpy


class Model(str, Enum):
    """
    Type of model used to train the network.
    """

    LAGRANGE = 'lagrange'
    LATENT_EQUILIBRIUM = 'latent_equilibrium'

class BackendType(str, Enum):
    """
    Type of computational backend to use for the lagrangian.
    """

    TENSORFLOW = 'tf'
    PYTORCH = 'torch'
    PYTORCH_BATCHED = 'torch_batched'
    NUMPY = 'numpy'
    JAX = 'jax'

class ModelVariant(str, Enum):
    """
    Which variant of the model to use. Default is predictive coding scheme but the lookahead
    voltages can also be updated directly without the use of differential equations.
    """

    # LE variants
    VANILLA = 'vanilla'
    INSTANTANEOUS = 'instantaneous'
    FULL_FORWARD_PASS = 'full_forward_pass'

    # BP implementation for sanity checks
    BACKPROP = 'backprop'

class Solver(str, Enum):
    """
    Type of solver to get the voltage gradients in case of the NLA model.
    """

    # NLA solvers
    CHOLESKY = 'cholesky'
    LU = 'lu'
    LS = 'ls'

    # LE solvers
    EULER = 'euler'  # iteraively update voltage derivative using euler steps

class ArchType(str, Enum):
    """
    Type of connection architecture for the network.
    """

    LAYERED_FEEDFORWARD = 'layered-feedforward'                                 # make fully-connected layers from input neurons to output neurons (classical feedforward NN)
    LAYERED_FEEDBACKWARD = 'layered-feedbackward'                               # make fully-connected layers from output neurons to input neurons (special case FFNN)
    LAYERED_RECURRENT = 'layered-recurrent'                                     # make fully-connected layers from input neurons to output neurons, and inverse
    LAYERED_RECURRENT_RECURRENT = 'layered-recurrent-recurrent'                 # make fully-connected layers from input neurons to output neurons, each layer recurrent
    FULLY_RECURRENT = 'fully-recurrent'                                         # make fully-connected network of neurons, except of self-connections
    IN_RECURRENT_OUT = 'in-recurrent-out'                                       # make input layer, recurrent hidden layers and output layer

class ActivationFunction(str, Enum):
    """
    Activation function for the neurons.
    """

    SIGMOID = 'sigmoid'                                                         # sigmoidal activation function
    MOD_SIGMOID = 'mod_sigmoid'                                                 # modified sigmoid with slope of ~ 1 and shifted to the right by 0.5
    HARD_SIGMOID = 'hard_sigmoid'                                               # hard sigmoid activation function
    RELU = 'relu'                                                               # ReLU activation function
    TANH = 'tanh'                                                               # Tangens hyperbolicus activation function
    ELU = 'elu'                                                                 # ELU activation function
    SWISH = 'swish'                                                             # Swish activation function

class TargetType(str, Enum):
    """
    Use either rate-based or voltage-based target errors in the case of LE.
    """

    RATE = 'rate'                                                               # use output rates to define the target error
    VOLTAGE = 'voltage'                                                         # use lookahead voltages to define the target error

class NetworkParams:
    """
    LE & NLA network parameters class. It allows to save to and load from json.
    Instances of this NetworkParams class defines common parameters of both
    the NLA & LE model as well as model specific parameters.
    """

    # default model and backend
    model = Model.LATENT_EQUILIBRIUM
    backend = BackendType.NUMPY

    layers = [2, 10, 10, 1]                                         # layer structure
    learning_rate_factors = [1, 1, 1]                               # learning rate factor to scale learning rates for each layer
    arg_tau = 10.0                                                  # time constant
    arg_tau_m = 10.0                                                # membrane time constant
    arg_tau_s = 0.0                                                 # synaptic time constant
    arg_dt = 0.1                                                    # integration step
    arg_beta = 0.1                                                  # nudging parameter beta

    arg_lrate_W = 0.1                                               # learning rate of the network weights
    arg_lrate_B = 0.0                                               # learning rate of feedback weights B
    arg_lrate_biases = 0.1                                          # learning rate of the biases

    arg_w_init_params = {'mean': 0, 'std': 0.1, 'clip': 0.3}        # weight initialization params for normal distribution sampling (mean, std, clip)
    arg_b_init_params = {'mean': 0, 'std': 0.1, 'clip': 0.3}        # bias initialization params for normal distribution sampling(mean, std, clip)

    arg_noise_width = 0.0                                           # if not zero, white noise is added to the firing rates at each timestep

    use_biases = True                                               # False: use model neurons with weights only, True: use biases+weights

    activation_function = ActivationFunction.SIGMOID                # change activation function, currently implemented: sigmoid, ReLU, capped ReLU
    network_architecture = ArchType.LAYERED_FEEDFORWARD             # network architecture used (defines the connectivity pattern / shape of weight matrix)
    #        currently implemented: LAYERED_FORWARD (FF network),
    #                               LAYERED_RECURRENT (every layer is recurrent)
    #                               FULLY_RECURRENT (completely recurrent),
    #                               IN_RECURRENT_OUT (input layer - layers of recurr. networks - output layer)

    is_timeseries = True                                            # if input neurons receive any inputs through synaptic connections
    only_discriminative = False                                     # if the network is not used as a generative model, then the input layers needs no biases and no incoming connections
    feedback_alignment = False                                      # use fixed backward weights

    dtype = numpy.float32                                           # data type used in calculations

    rnd_seed = 42                                                   # random seed

    # LE specific parameters
    target_type = TargetType.VOLTAGE                                # use voltages or rates for target error
    model_variant = ModelVariant.VANILLA                            # variant of the model
    integration_method = Solver.EULER                               # numerical integration method

    # NLA specific parameters
    solver_type = Solver.LU  # use CHOLESKY solver (faster, might fail due to numerical issues) or LU solver (slower, but works always)

    arg_clip_weight_deriv = False  # weight derivative update clipping
    arg_weight_deriv_clip_value = 0.05  # weight derivative update clipping

    use_sparse_mult = False  # use sparse multiplication when calculating matrix product of two matrices
    only_discriminative = True  # if the network is not used as a generative model, then the visible layers needs no biases
    turn_off_visible_dynamics = False  # either clamp or nudge the visible layer to the input

    check_with_profiler = False  # run graph once with tf-profiler on
    write_tensorboard = False

    # interneuron parameters
    arg_lrate_biases_I = 0.1  # learning rate of interneuron biases
    arg_lrate_PI = 0.  # learning rate of interneuron to principal weights W_PI
    arg_interneuron_b_scale = 1  # interneuron backward weight scaling
    use_interneurons = False  # train interneuron circuit (True) or use weight transport (False)
    dynamic_interneurons = False  # either interneuron voltage is integrated dynamically (True) or set to the stat. value instantly (False, =ideal theory)

    on_cuda = False  # run on cuda if available

    def __init__(self, file_name=None):
        if file_name is not None:
            self.load_params(file_name)

    def load_params(self, file_name):
        """
        Load parameters from json file.
        """
        with open(file_name, 'r') as file:
            deserialize_dict = json.load(file)
            for key, value in deserialize_dict.items():
                if isinstance(value, str) and 'numpy' in value:
                    value = eval(value)
                elif isinstance(value, str) and getattr(self, key).__class__ is not str:
                    key_class = getattr(self, key).__class__
                    value = key_class(value)

                setattr(self, key, value)

    def load_params_from_dict(self, dictionary):
        """
        Load parameters from dictionary.
        """
        for key, value in dictionary.items():
            # check if key is actually an attribute of the NetworkParams class
            if not hasattr(self, key):
                continue
            if isinstance(value, str) and 'numpy' in value:
                value = eval(value)
            elif isinstance(value, str) and getattr(self, key).__class__ is not str:
                key_class = getattr(self, key).__class__
                value = key_class(value)

            setattr(self, key, value)

    def save_params(self, file_name):
        """
        Save parameters to json file.
        """
        with open(file_name, 'w') as file:
            file.write(self.to_json())

    def to_json(self):
        """
        Turn network params into json.
        """
        serialize_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('__') and not callable(key):
                if callable(value):
                    if value is numpy.float32 or value is numpy.float64:
                        value = re.search('\'(.+?)\'', str(value)).group(1)
                    else:
                        break
                serialize_dict[key] = value

        return json.dumps(serialize_dict, indent=4)

    def __str__(self):
        """
        Return string representation.
        Returns:

        """
        return self.to_json()
