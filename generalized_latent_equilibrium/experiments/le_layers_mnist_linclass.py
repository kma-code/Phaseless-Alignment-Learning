# loads a trained model and trains a linear classifier on it

import argparse
import os
import io
import logging
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import model.latent_equilibrium_layers as nn
import model.layered_torch_utils as tu
from model.network_params import ModelVariant, TargetType

# for reproducibility inits
import random
import numpy as np
import pickle

import matplotlib.patches as mpatches
import pylab as plt
# plt.rc('text', usetex=True)
# plt.rc('font', size=12,family='serif')
# plt.style.use('matplotlibrc')
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
colors = ["r","g","b","black","yellow","gray","pink","cyan","magenta","lightblue"]


# give this to each dataloader
def dataloader_seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# custom unpickler for torch models
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

# cosine similarity between tensors
def cos_sim(A, B):
    if A.ndim == 1 and B.ndim == 1:
        return A.T @ B / np.linalg.norm(A) / np.linalg.norm(B)
    else:
        return np.trace(A.T @ B) / np.linalg.norm(A) / np.linalg.norm(B)

def deg(cos):
    # calculates angle in deg from cosine
    return np.arccos(cos) * 180 / np.pi

# networks


def linclass_test(model, train_loader, test_loader):

    logging.basicConfig(format='Running linear classifier model -- %(levelname)s: %(message)s',
                    level=logging.INFO, force=True)
    logging.info("Starting testing")

    # testing
    # correct_cnt = 0
    test_loss = []
    summed_error = 0
    total_cnt = 0
    latent_target_train_arr = []
    model.eval()

    for batch_idx, (x, target) in enumerate(train_loader):
        # we need to flatten for MNIST and MLPNet
        x = x.reshape([batch_size, 28 * 28])
        # we use an autoencoder
        # target = x
        if use_cuda:
            # x, target = x.cuda(), target.cuda()
            x = x.cuda()

        for update_i in range(presentation_steps):
            model.update(x, x)

        # latent activation of encoder/decoder is middle hidden layer
        latent = model.rho[int(len(model.layers)/2)]
        out = model.rho[-1]
        # during validation, loss is calculated from MSE of output vs target
        error = (x - out).mean()
        # _, pred_label = torch.max(out, 1)
        total_cnt += x.shape[0]
        # correct_cnt += (pred_label == torch.max(target, 1)[1]).sum()
        summed_error += error.detach().cpu().numpy()

        # we record target and associated latent activation
        latent_target_train_arr.append([latent.detach().cpu().tolist(), target.detach().cpu().tolist()])

    # convert form of latent_target array to usable list [latent, target]
    latent_target_train = []
    for batch_idx in range(len(latent_target_train_arr)):
        for sample_idx in range(len(latent_target_train_arr[batch_idx][0])):
            latent_target_train.append([latent_target_train_arr[batch_idx][0][sample_idx], latent_target_train_arr[batch_idx][1][sample_idx]])

    test_loss.append(np.abs(summed_error)/total_cnt)
    with open(PATH_OUTPUT + "latent_target_train_epoch" + str(model.epoch) + ".pkl", "wb") as output:
        pickle.dump(latent_target_train, output)
        logging.info(f"Saving latent and target activation to {output.name}")

    # # generate a plot of latent activity
    # ax = plt.figure(figsize=(7,5))
    # for latent, target in latent_target_train[:1000]:
    #     lab_pos = np.where(np.array(target)==1)[0][0]
    #     plt.plot(latent[0], latent[1], c=colors[lab_pos], marker='o')
    # # plt.scatter(*zip(*[latent_target_train[0] for latent_target_train in latent_target_train]))

    # for color, label in zip(colors, range(10)):
    #     plt.scatter([],[], marker="o", c=color, label=str(label))
        
    # ax.legend(loc='center left', bbox_to_anchor=(.9, 0.5))

    # IMG_NAME = PATH_OUTPUT + "latent_activity_epoch" + str(model.epoch) + ".png"

    # logging.info(f"Saving image of latent activation to {IMG_NAME}")
    # plt.savefig(IMG_NAME)

    # # plot the last image currently fed into the model
    # ax = plt.figure(figsize=(3,3))
    # input_data = x[-1].detach().cpu().numpy()
    # input_data = input_data.reshape(28,28)
    # plt.imshow(input_data, cmap='gray', interpolation='none')
    # plt.axis('off')
    # IMG_NAME = PATH_OUTPUT + "example_input" + str(model.epoch) + ".png"
    # logging.info(f"Saving image of last input to {IMG_NAME}")
    # plt.savefig(IMG_NAME)

    # # plot the last image currently saved in model
    # ax = plt.figure(figsize=(3,3))
    # output_data = model.rho[-1][-1].detach().cpu().numpy()
    # output_data = output_data.reshape(28,28)
    # plt.imshow(output_data, cmap='gray', interpolation='none')
    # plt.axis('off')
    # IMG_NAME = PATH_OUTPUT + "example_output" + str(model.epoch) + ".png"
    # logging.info(f"Saving image of last output to {IMG_NAME}")
    # plt.savefig(IMG_NAME)



    # now, train a linear classifier on latent space (train set) and evaluate it using the test set
    logging.info(f"Training linear classifier")
    # make arrays of only latent vecs and only targets
    latent_vecs_train = [latent_target_train[0] for latent_target_train in latent_target_train]
    targets_train = [latent_target_train[1] for latent_target_train in latent_target_train]
    # convert targets from one-hot to integers
    targets_train = [np.where(np.array(vec)==1)[0][0] for vec in targets_train]

    clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=1000, tol=1e-3))
    clf.fit(latent_vecs_train, targets_train)




    logging.info(f"Predicting with linear classifier on test set")

    latent_target_test_arr = []

    for batch_idx, (x, target) in enumerate(test_loader):
        # we need to flatten for MNIST and MLPNet
        x = x.reshape([batch_size, 28 * 28])
        # we use an autoencoder
        # target = x
        if use_cuda:
            # x, target = x.cuda(), target.cuda()
            x = x.cuda()

        for update_i in range(presentation_steps):
            model.update(x, x)

        # latent activation of encoder/decoder is middle hidden layer
        latent = model.rho[int(len(model.layers)/2)]
        out = model.rho[-1]
        # during validation, loss is calculated from MSE of output vs target
        error = (x - out).mean()
        # _, pred_label = torch.max(out, 1)
        total_cnt += x.shape[0]
        # correct_cnt += (pred_label == torch.max(target, 1)[1]).sum()
        summed_error += error.detach().cpu().numpy()

        # we record target and associated latent activation
        latent_target_test_arr.append([latent.detach().cpu().tolist(), target.detach().cpu().tolist()])

    # convert form of latent_target array to usable list [latent, target]
    latent_target_test = []
    for batch_idx in range(len(latent_target_test_arr)):
        for sample_idx in range(len(latent_target_test_arr[batch_idx][0])):
            latent_target_test.append([latent_target_test_arr[batch_idx][0][sample_idx], latent_target_test_arr[batch_idx][1][sample_idx]])


    # make arrays of only latent vecs and only targets
    latent_vecs_test = [latent_target_test[0] for latent_target_test in latent_target_test]
    targets_test = [latent_target_test[1] for latent_target_test in latent_target_test]
    # convert targets from one-hot to integers
    targets_test = [np.where(np.array(vec)==1)[0][0] for vec in targets_test]

    total_cnt = 0
    correct_cnt = 0
    for latent, target in zip(latent_vecs_test, targets_test):
        pred = clf.predict([latent])[0]
        if pred == target:
            correct_cnt += 1
        total_cnt += 1
    logging.info(f"Accuracy: {correct_cnt/total_cnt}")

    return correct_cnt/total_cnt

if __name__ == '__main__':

    logging.basicConfig(format='Model setup -- %(levelname)s: %(message)s',
                    level=logging.INFO, force=True)

    parser = argparse.ArgumentParser(description='Train a Latent Equilibrium Neural Network on MNIST and check accuracy.')
    parser.add_argument('--algorithm', default="BP", type=str, help="Model algorithm: BP or FA")
    parser.add_argument('--model_variant', default="vanilla", type=str, help="Model variant: vanilla, full_forward_pass")
    parser.add_argument('--batch_size', default=32, type=int, help="Batch size for training")
    parser.add_argument('--batch_learning_multiplier', default=64, type=int, help="Learning rate multiplier for batch learning")
    parser.add_argument('--lr_factors', default=[1e-3,1e-3,1e-3,1e-3], help="Learning rate multipliers for each layer")
    parser.add_argument('--n_updates', default=10, type=int, help="Number of update steps per sample/batch (presentation time in time steps dt)")
    parser.add_argument('--with_optimizer', action='store_true', help="Train network with Adam Optimizer")
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to train.')
    parser.add_argument('--seed', default=7, type=int, help='Seed for reproducibility.')
    parser.add_argument('--load', default=None, type=str, help='Saved network to load.')
    parser.add_argument('--params', default=None, type=str, help='Load parameters from file (overrides manual params).')
    parser.add_argument('--wn_sigma', default=[0,0,0.3,0.3], help="Stdev of white noise injected into each layer")
    # additional params for PAL
    parser.add_argument('--bw_lr_factors', default=[1e-2,1e-2,1e-2,1e-2], help="Learning rate multipliers for backwards weights originating from each layer")
    parser.add_argument('--regularizer', default=[1e-4,1e-4,1e-4,1e-4], help="Size of weight decay regularizer")
    parser.add_argument('--tau_xi', default=[10,10,10,10], help="Filter constant of Ornstein-Uhlenbeck noise (given in time steps dt)")
    parser.add_argument('--tau_HP', default=[10,10,10,10], help="Time constant of high-pass filter (given in time steps dt)")
    parser.add_argument('--tau_LO', default=[1e+4,1e+4,1e+4,1e+4], help="Time constant of low-pass filter in forward weights (given in time steps dt)")
    parser.add_argument('--sigma', default=[1e-2,1e-2,1e-2,0], help="Stdev of Ornstein-Uhlenbeck noise injected into each layer")
    # recording of params
    parser.add_argument('--rec_weights', default=False, action='store_true', help="Record all weights of net")
    parser.add_argument('--rec_activations', default=False, action='store_true', help="Record activations after every presentation time")
    parser.add_argument('--rec_noise', default=False, action='store_true', help="Record injected noise (PAL only)")


    args = parser.parse_args()
    # path to folder of this file
    PATH_SCRIPT = os.path.dirname(os.path.realpath(__file__))
    PATH_PARAMS = os.path.dirname(os.path.realpath(args.params)) + '/'

    if args.params is not None:
        with open(args.params, 'r+') as f:
            PARAMETERS = json.load(f)

        seed = PARAMETERS["seed"]
        batch_size = PARAMETERS["batch_size"]
        lr_multiplier = PARAMETERS["batch_learning_multiplier"]
        lr_factors = PARAMETERS["lr_factors"]
        algorithm = PARAMETERS["algorithm"]
        model_variant = PARAMETERS["model_variant"]
        target_type = TargetType.RATE
        presentation_steps = PARAMETERS["n_updates"]
        epochs = PARAMETERS["epochs"]
        tqdm_disabled = True
        PATH_OUTPUT = PARAMETERS["output"]

        if "rec_weights" in PARAMETERS:
            rec_weights = PARAMETERS["rec_weights"]
        else:
            rec_weights = False
        if "rec_activations" in PARAMETERS:
            rec_activations = PARAMETERS["rec_activations"]
        else:
            rec_activations = False
        if "rec_noise" in PARAMETERS:
            rec_noise = PARAMETERS["rec_noise"]
        else:
            rec_noise = False

        if algorithm == 'PAL':
            bw_lr_factors = PARAMETERS["bw_lr_factors"]
            regularizer = PARAMETERS["regularizer"]
            tau_xi = PARAMETERS["tau_xi"]
            tau_HP = PARAMETERS["tau_HP"]
            tau_LO = PARAMETERS["tau_LO"]
            sigma = PARAMETERS["sigma"]
        wn_sigma = PARAMETERS["wn_sigma"]

    else:
        PATH_OUTPUT = PATH_SCRIPT + '/output/'

        seed = args.seed
        batch_size = args.batch_size
        lr_multiplier = args.batch_learning_multiplier
        lr_factors = args.lr_factors
        algorithm = args.algorithm
        model_variant = args.model_variant
        target_type = TargetType.RATE
        presentation_steps = args.n_updates
        epochs = args.epochs
        with_optimizer = args.with_optimizer
        tqdm_disabled = False

        bw_lr_factors = args.bw_lr_factors
        regularizer = args.regularizer
        tau_xi = args.tau_xi
        tau_HP = args.tau_HP
        tau_LO = args.tau_LO
        sigma = args.sigma
        wn_sigma = args.wn_sigma

        rec_weights = args.rec_weights
        rec_activations = args.rec_activations
        rec_noise = args.rec_noise

    with_optimizer = False

    logging.info(f"Params: Epochs {epochs}, batch_size {batch_size}, lr_multiplier {lr_multiplier}, lr_factors {lr_factors}, white noise {wn_sigma}, algorithm {algorithm}")
    # if algorithm == "PAL":
    #     logging.info(f"Params: bw_lr_factors {bw_lr_factors}, regularizer {regularizer}, tau_xi {tau_xi}, tau_HP {tau_HP}, tau_LO {tau_LO}, sigma {sigma}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # setup network parameters
    tau = 10.0
    dt = 0.1
    beta = 0.1


    # load MNIST dataset
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        logging.info("Cuda enabled")
    else:
        logging.info("Cuda disabled")

    # We exclude any data augmentation here that generates more training data
    transform = transforms.Compose([transforms.ToTensor()])# , transforms.Normalize((0.1307,), (0.3081,))]) # normalize doesn't improve auto-enc results
    target_transform = transforms.Compose([
        lambda x:torch.LongTensor([x]),
        lambda x: F.one_hot(x, 10),
        lambda x: x.squeeze()
    ])

    # if not existing, download mnist dataset
    train_set = datasets.MNIST(root=PATH_SCRIPT + '/mnist_data', train=True, transform=transform, target_transform=target_transform, download=True)
    test_set  = datasets.MNIST(root=PATH_SCRIPT + '/mnist_data', train=False, transform=transform, target_transform=target_transform, download=True)

    val_size = 10000
    train_size = len(train_set) - val_size

    train_set, val_set = random_split(train_set, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    num_workers = 0
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, worker_init_fn=dataloader_seed_worker, drop_last=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, worker_init_fn=dataloader_seed_worker, drop_last=True)

    logging.info(f'Total training batches: {len(train_loader)}')
    logging.info(f'Total testing batches: {len(test_loader)}')

    if rec_weights:
        logging.info(f'Recording weights after every evaluation: {rec_weights}')
        weights_time_series = []
        bw_weights_time_series = []
    if rec_activations or rec_noise:
        logging.info(f'Recording during evaluation after every presentation time: Weights: {rec_weights}, Activations: {rec_activations}, Noise: {rec_noise}')

    # array to register linear classifier accuracy
    lin_acc_arr = []

    for i in range(epochs + 1):
        logging.info(f"Working on epoch {i}")

        # load model from same folder as params file

        MODEL = 'MLPNet_epoch' + str(i) + '.pkl'
        with open(PATH_PARAMS + MODEL, 'rb') as input:
            if use_cuda  == False:
                model = CPU_Unpickler(input).load()
                # we also need to set all device variables to cpu
                for layer in model.layers:
                    layer.device = 'cpu'
                model.device = 'cpu'
            else:
                model = pickle.load(input)
            logging.info(f"Loaded model {MODEL}")

        # evaluate model on test set

        lin_acc = linclass_test(model, train_loader, test_loader)
        lin_acc_arr.append(lin_acc)

    logging.info("Saving linear classifier accuracy.")
    np.save(PATH_PARAMS + "lin_acc.npy", lin_acc_arr)




