
import random
import numpy as np
from pathlib import Path

from model.network_params import ActivationFunction


def get_digits_from_class(images, labels, target_size: int, digit_class: int, samples_qty: int, activation_is_sigmoid: bool = False, reshape=False):
    """
    Get samples of a certain digit class of either the training set or test set.

    Args:
    images: images of dataset
    labels: labels of dataset
    target_size: size of the output
    digit_class: class [1-10] of digit
    samples_qty: number of samples for this digit class (-1 means getting all samples available)
    activation_is_sigmoid: if the data should be prepared for sigmoidal activation neurons

    Returns:

    """

    # one hot encode digit class
    label = np.zeros(target_size)  # label must be of size of the last neuronal network layer
    label[digit_class] = 1.0

    if samples_qty == -1:  # -1 = get all samples
        samples_qty = len(list(images[labels == str(digit_class)]))

    # get number of samples from MNIST
    digits_255 = random.sample(list(images[labels == str(digit_class)]), samples_qty)  # get number of random samples from all images which belong to the indicated digit class
    digits_255 = [digit_255.astype(np.float32) for digit_255 in digits_255]  # convert to float
    digits = [np.multiply(digit_255, 1.0 / 255.0) for digit_255 in digits_255]  # scale digits representation to 0-1
    dataset = [[digit, label] for digit in digits]  # make M x 1 array and combine with label

    # transform data if activation is sigmoid to reduce vanishing gradient problems
    if activation_is_sigmoid:
        for i in range(samples_qty):
            scaled_digit = (dataset[i][0] + 0.1) / 1.1 * 0.9
            scaled_label = (dataset[i][1] + 0.1) / 1.1 * 0.9
            transformed_digit = np.log(scaled_digit / (1 - scaled_digit))
            transformed_label = np.log(scaled_label / (1 - scaled_label))
            dataset[i][0] = transformed_digit
            dataset[i][1] = transformed_label

    if reshape:
        for data in dataset:
            data[0] = data[0].reshape(reshape)

    return dataset


def transform_with_activation(x, activation_type: ActivationFunction):
    if activation_type == ActivationFunction.RELU:
        return x * (x > 0)
    elif activation_type == ActivationFunction.SIGMOID:
        return 1. / (1 + np.exp(-np.clip(x, -10, 10)))
    elif activation_type == ActivationFunction.HARD_SIGMOID:
        return np.clip(x, 0, 1)
    else:
        raise Exception("Unknown type of activation.")


def decay_func(x, equi1, equi2, tau):
    return equi2 + (equi1 - equi2) * np.exp(-x / tau)


def save_params(network, epoch_i, base_save_path="./checkpoints"):
    """
    Save current params of the network.
    Args:
        network: the lagrangian network.
        epoch_i: the current training epoch.
        base_save_path: base save path for saving the parameters.

    """
    save_path = Path(base_save_path + "/network_params_e" + str(epoch_i))
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    network.save(save_path)


def load_params(network, epoch_i, base_load_path="./checkpoints", dry_run=False):
    """
    Load current params of the network from file.
    Args:
        network:
        epoch_i:
        base_load_path:
        dry_run:

    Returns:

    """

    load_path = Path(base_load_path + "/network_params_e" + str(epoch_i))
    if not load_path.exists():
        raise Exception("{0} does not exist. Loading network aborted.".format(load_path))
    elif dry_run:
        return True

    return network.load(load_path)
