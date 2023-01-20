import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


### General Utils ###

def set_tensor(xs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return xs.float().to(device)


### Activation functions ###

class TanH:
    @staticmethod
    def f(xs):
        return torch.tanh(xs)

    @staticmethod
    def df(xs):
        return 1.0 - torch.tanh(xs) ** 2.0


class Linear:
    @staticmethod
    def f(x):
        return x

    @staticmethod
    def df(x):
        return set_tensor(torch.ones((1, x.shape[1])))


class ReLU:
    @staticmethod
    def f(xs):
        return torch.clamp(xs, min=0)

    @staticmethod
    def df(xs):
        rel = ReLU.f(xs)
        rel[rel > 0] = 1
        return rel


class HardSigmoid:
    @staticmethod
    def f(xs):
        return xs.clamp(0, 1)

    @staticmethod
    def df(xs):
        return set_tensor((xs >= 0) * (xs <= 1))


class Sigmoid:
    @staticmethod
    def f(xs):
        return F.sigmoid(xs)

    @staticmethod
    def df(xs):
        return F.sigmoid(xs) * (torch.ones_like(xs) - F.sigmoid(xs))


def softmax(xs):
    return F.softmax(xs)


### loss functions
def mse_loss(out, label):
    return torch.sum((out - label) ** 2)


def mse_deriv(out, label):
    return 2 * (out - label)


ce_loss = nn.CrossEntropyLoss()


def cross_entropy_loss(out, label):
    return ce_loss(out, label)


def my_cross_entropy(out, label):
    return -torch.sum(label * torch.log(out + 1e-6))


def cross_entropy_deriv(out, label):
    return out - label


def parse_loss_function(loss_arg):
    if loss_arg == "mse":
        return mse_loss, mse_deriv
    elif loss_arg == "crossentropy":
        return my_cross_entropy, cross_entropy_deriv
    else:
        raise ValueError("loss argument not expected. Can be one of 'mse' and 'crossentropy'. You input " + str(loss_arg))


### Initialization Functions ###
def gaussian_init(W, mean=0.0, std=0.05):
    return W.normal_(mean=0.0, std=0.05)


def zeros_init(W):
    return torch.zeros_like(W)


def kaiming_init(W, a=math.sqrt(5), *kwargs):
    return init.kaiming_uniform_(W, a)


def glorot_init(W):
    return init.xavier_normal_(W)


def kaiming_bias_init(b, *kwargs):
    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1 / math.sqrt(fan_in)
    return init.uniform_(b, -bound, bound)


# the initialization pytorch uses for lstm
def std_uniform_init(W, hidden_size):
    stdv = 1.0 / math.sqrt(hidden_size)
    return init.uniform_(W, -stdv, stdv)
