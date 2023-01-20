#!/usr/bin/env python3
# Wrapper over different models and implementations for easier comparisons.
#
# Authors: Paul Haider

# implementation independent network parameters
from model.network_params import BackendType, Model, ModelVariant, NetworkParams, Solver

# warn about usage of some method
from model.dev_utils import internal


class NetworkWrapper:
    """
    Wrapper over all implementations to make it easier to switch between them.
    """

    def __init__(self, model : Model, backend: BackendType, params: NetworkParams):
        # store model params
        self.model = model
        self.backend = backend

        # print model params
        self.report_model_params()

        # choose appropriate model
        if model == Model.LATENT_EQUILIBRIUM:
            if backend == BackendType.NUMPY:
                import model.latent_equilibrium_np as le_net
                self.network = le_net.LatentEqNetwork(params)
            elif backend == BackendType.PYTORCH:
                import model.latent_equilibrium_torch as le_net
                self.network = le_net.LatentEqNetwork(params)
            elif backend == BackendType.PYTORCH_BATCHED:
                import model.latent_equilibrium_torch_batched as le_net
                self.network = le_net.LatentEqNetwork(params)
            # raise error if backend is not numpy
            else:
                raise NotImplementedError(f"Backend type '{backend}' not implemented for model '{model}'")

        elif model == Model.LAGRANGE:
            if backend == BackendType.TENSORFLOW:
                import model.lagrange_model_tf as lg_tf
                self.network = lg_tf.LagrangeNetwork(params)
            elif backend == BackendType.PYTORCH:
                import model.lagrange_model_torch as lg_torch
                self.network = lg_torch.LagrangeNetwork(params)
            elif backend == BackendType.PYTORCH_BATCHED:
                import model.lagrange_model_torch_batched as lg_torch
                self.network = lg_torch.LagrangeNetwork(params)
            else:
                raise NotImplementedError(f"Backend type '{backend}' not implemented for model '{model}'")
        else:
            raise NotImplementedError(f"Model type '{model}' not implemented")

        # define dictionary for logging functionaly, in particular in combination with sacred
        self.logs = {}

    def set_beta(self, beta):
        """
        Set the nudging beta.

        Args:
            beta:

        Returns:

        """
        self.network.set_beta(beta)

    def report_model_params(self):
        """
        Print report of model setup to the terminal.
        """
        print('------------------')
        print('MODEL')
        print('------------------')
        print('Model: ', self.model)
        print('Backend: ', self.backend)

    def report_simulation_params(self):
        """
        Print report of simulation setup to the terminal.
        """
        self.network.report_simulation_params()

    # NETWORK INTEGRATOR
    ######################
    def update_network(self, input_voltage, output_voltage, train_W=False, train_B=False, train_PI=False, record_observables=False):
        """
        Perform a single integration step.
        """
        self.network.update_network(input_voltage, output_voltage, train_W, train_B, train_PI, record_observables)

    # GET STATES
    def get_voltages(self):
        return self.network.get_voltages()

    def get_voltage_derivatives(self):
        return self.network.get_voltage_derivatives()

    def get_errors(self):
        return self.network.get_errors()

    def get_error_lookaheads(self):
        return self.network.get_error_lookaheads()

    def get_weights(self):
        return self.network.get_weights()

    def get_biases(self):
        return self.network.get_biases()

    # SAVE AND LOAD NETWORK
    def save(self, save_path):
        """
        Save the lagrangian model to file.
        Args:
            save_path:

        Returns:

        """
        self.network.save(save_path)

    def load(self, load_path):
        """
        Load the lagrangian model from file.
        Note: Not yet meant to be cross-compatible!

        Args:
            load_path:

        Returns:

        """
        return self.network.load(load_path)

    # GET and SET
    def set_lrate_W(self, learning_rate):
        self.network.set_lrate_W(learning_rate)

    def set_lrate_B(self, learning_rate):
        self.network.set_lrate_B(learning_rate)

    def set_lrate_biases(self, learning_rate):
        self.network.set_lrate_biases(learning_rate)

    def set_lrate_biases_I(self, learning_rate):
        self.network.set_lrate_biases_I(learning_rate)

    def set_weights(self, weights):
        self.st_weights = weights

    def get_lrate_W(self):
        return self.network.arg_lrate_W

    def get_lrate_B(self):
        return self.network.arg_lrate_B

    def get_lrate_biases(self):
        return self.network.arg_lrate_biases

    # KERAS-like INTERFACE
    def fit(self, x=None, y=None, n_updates: int=100, batch_size=1, epochs: int=1, verbose: int=1, is_timeseries: bool=False):
        """
        Train network on dataset.

        Args:
            x: dataset of samples to train on.
            y: respective labels to train on.
            n_updates: number of weight updates per sample.
            batch_size: Number of examples per batch (batch training).
            nudging_beta: Amount of nudging applied to the network.
            epochs: Amount of epochs to train for.
            verbose: Level of verbosity.
            is_timeseries: if input is already a timeseries or not
        """

        self.network.fit(x, y, n_updates, batch_size, epochs, verbose, is_timeseries)
        # copy train logs to wrapper
        for key in self.network.logs:
            self.logs[key] = self.network.logs[key]

    def fit_batch(self, x=None, y=None, n_updates: int=100, batch_iteration=-1, batch_qty=-1, verbose: int=1, is_timeseries: bool=False):
        self.network.fit_batch(x, y, n_updates, batch_iteration, batch_qty, verbose, is_timeseries)

    def predict(self, x, n_updates: int = 100, batch_size=1, verbose: int=1, is_timeseries: bool=False):
        """
        Predict batch with trained network.

        Args:
            batch_size:
            x: samples to be predicted.
            n_updates: number of updates of the network used in tests.
            batch_size: Number of examples per batch (batch training).
            verbose: Level of verbosity.
            is_timeseries: Is the input a time series already or not.

        Returns:
            predictions: array of predictions for samples x
        """

        return self.network.predict(x, n_updates, batch_size, verbose, is_timeseries)

    def predict_batch(self, x, n_updates: int = 100, batch_iteration=-1, batch_qty=-1, verbose: int=1, is_timeseries: bool=False):
        return self.network.predict_batch(x, n_updates, batch_iteration, batch_qty, verbose, is_timeseries)

    def __call__(self, x, n_updates: int = 100, batch_iteration=-1, batch_qty=-1, verbose: int=1, is_timeseries: bool=False):
        return self.predict_batch(x, n_updates, batch_iteration, batch_qty, verbose, is_timeseries)

    @internal
    def get_internal_network(self):
        """
        Returns the internal network to modify it. This should only be used temporarily, thus the deprecation warning.
        Returns:

        """
        return self.network

    @property
    def layers(self):
        return self.network.layers

    # HELPER FUNCTIONS IN NOTEBOOKS
    # TODO: Replace the usage of this method with its torch equivalent
    ###############################
    @internal
    def get_rates(self, voltages, voltages_deriv):
        """
        Returns rate+rate lookahead for act. function. Purely used as a utility function!
        """
        return self.network.get_rates(voltages, voltages_deriv)

