# reference implementation of MNIST training

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

def MLPNet(batch_size, lr_multiplier, lr_factors, tau=10.0, dt=0.1, beta=0.1, algorithm='BP',
           model_variant=ModelVariant.VANILLA, target_type=TargetType.RATE, presentation_steps=10, with_optimizer=False,
           bw_lr_factors=None, regularizer=None, tau_xi=None, tau_HP=None, tau_LO=None, sigma=None, wn_sigma=[0,0,0,0]):
    """
    4 layer fully connected network
    """
    learning_rate = 0.125 * lr_multiplier / presentation_steps / dt

    act_func = tu.TanH
    logging.info(f"Initializing network using {algorithm}")

    if algorithm == 'PAL':
        # architecture following 1906.00889
        # encoder
        fc1 = nn.Linear_PAL(28 * 28, 200, act_func, algorithm=algorithm)
        fc2 = nn.Linear_PAL(200, 2, tu.Linear, algorithm=algorithm)
        # decoder
        fc3 = nn.Linear_PAL(2, 200, act_func, algorithm=algorithm)
        fc4 = nn.Linear_PAL(200, 28 * 28, tu.Linear, algorithm=algorithm)

        network = nn.LESequential([fc1, fc2, fc3, fc4], learning_rate, lr_factors, None, None,
                                  tau, dt, beta, model_variant, target_type, with_optimizer=with_optimizer, algorithm=algorithm,
                                  bw_lr_factors=bw_lr_factors, regularizer=regularizer, tau_xi=tau_xi, tau_HP=tau_HP, tau_LO=tau_LO, sigma=sigma, wn_sigma=wn_sigma)

        for layer in network.layers:
            logging.info(f"Initialized layer with PAL parameters {layer.get_PAL_parameters()}")

    else:
        # architecture following 1906.00889
        # encoder
        fc1 = nn.Linear(28 * 28, 200, act_func, algorithm=algorithm)
        fc2 = nn.Linear(200, 2, tu.Linear, algorithm=algorithm)
        # decoder
        fc3 = nn.Linear(2, 200, act_func, algorithm=algorithm)
        fc4 = nn.Linear(200, 28 * 28, tu.Linear, algorithm=algorithm)

        network = nn.LESequential([fc1, fc2, fc3, fc4], learning_rate, lr_factors, None, None,
                                  tau, dt, beta, model_variant, target_type, with_optimizer=with_optimizer, algorithm=algorithm, sigma=sigma, wn_sigma=wn_sigma)

    return network


def validate_model(model, val_loader):

    # validation
    correct_cnt, summed_error = 0, 0
    total_cnt = 0
    model.eval()

    for batch_idx, (x, target) in enumerate(val_loader):
        # we need to flatten for MNIST and MLPNet
        x = x.reshape([batch_size, 28 * 28])
        # we use an autoencoder
        # target = x
        if use_cuda:
            # x, target = x.cuda(), target.cuda()
            x = x.cuda()

        for update_i in range(presentation_steps):
            model.update(x, x)

        out = model.rho[-1]
        # during validation, loss is calculated from MSE of output vs target
        error = (x - out).mean()
        # _, pred_label = torch.max(out, 1)
        total_cnt += x.shape[0]
        # correct_cnt += (pred_label == torch.max(target, 1)[1]).sum()
        summed_error += error.detach().cpu().numpy()

        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(val_loader):
            if model.epoch == 0:
                logging.info(f'Epoch: before training, batch index: {batch_idx + 1}, val loss:  {np.abs(summed_error)/total_cnt:.9f}')
            else:
                logging.info(f'Epoch: {model.epoch}, batch index: {batch_idx + 1}, val loss:  {np.abs(summed_error)/total_cnt:.9f}')

    # record weights
    weights_arr = [layer.weights.detach().cpu().numpy() for layer in model.layers] if rec_weights else None
    if model.algorithm in ["FA", "PAL"]:
        bw_weights_arr = [layer.bw_weights.detach().cpu().numpy() for layer in model.layers] if rec_weights else None
    elif model.algorithm == 'BP':
        bw_weights_arr = [layer.weights.t().detach().cpu().numpy() for layer in model.layers] if rec_weights else None

    return np.abs(summed_error)/total_cnt, weights_arr, bw_weights_arr


def test_model(model, test_loader):

    logging.basicConfig(format='Testing model -- %(levelname)s: %(message)s',
                    level=logging.INFO, force=True)
    logging.info("Starting testing")

    # testing
    # correct_cnt = 0
    test_loss = []
    summed_error = 0
    total_cnt = 0
    # latent_target_arr = []
    model.eval()

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
        # latent = model.rho[int(len(model.layers)/2)]
        out = model.rho[-1]
        # during validation, loss is calculated from MSE of output vs target
        error = (x - out).mean()
        # _, pred_label = torch.max(out, 1)
        total_cnt += x.shape[0]
        # correct_cnt += (pred_label == torch.max(target, 1)[1]).sum()
        summed_error += error.detach().cpu().numpy()

        # we record target and associated latent activation
        # latent_target_arr.append([latent.detach().cpu().tolist(), target.detach().cpu().tolist()])

        # record activations
        # if rec_activations:
        #     activation_arr.append(model.rho.detach().cpu().tolist())

        # record noise
        # if rec_noise:
        #     noise_arr.append(model.noise.detach().cpu().tolist())

        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(val_loader):
            logging.info(f'Epochs trained: {model.epoch}, batch index: {batch_idx + 1}, test loss:  {np.abs(summed_error)/total_cnt:.9f}')

    # convert form of latent_target array to usable list [latent, target]
    # latent_target = []
    # for batch_idx in range(len(latent_target_arr)):
    #     for sample_idx in range(len(latent_target_arr[batch_idx][0])):
    #         latent_target.append([latent_target_arr[batch_idx][0][sample_idx], latent_target_arr[batch_idx][1][sample_idx]])

    # latent_target = torch.Tensor(latent_target)

    # test_loss.append(np.abs(summed_error)/total_cnt)
    # with open(PATH_OUTPUT + "latent_target_epoch" + str(model.epoch) + ".pkl", "wb") as output:
    #     pickle.dump(latent_target, output)
    #     logging.info(f"Saving latent and target activation to {output.name}")

    # generate a plot of latent activity
    # ax = plt.figure(figsize=(7,5))
    # for latent, target in latent_target[:1000]:
    #     lab_pos = np.where(np.array(target)==1)[0][0]
    #     plt.plot(latent[0], latent[1], c=colors[lab_pos], marker='o')
    # plt.scatter(*zip(*[latent_target[0] for latent_target in latent_target]))

    # for color, label in zip(colors, range(10)):
    #     plt.scatter([],[], marker="o", c=color, label=str(label))
        
    # ax.legend(loc='center left', bbox_to_anchor=(.9, 0.5))

    # IMG_NAME = PATH_OUTPUT + "latent_activity_epoch" + str(model.epoch) + ".png"

    # logging.info(f"Saving image of latent activation to {IMG_NAME}")
    # plt.savefig(IMG_NAME)

    # plot the last image currently fed into the model
    ax = plt.figure(figsize=(3,3))
    input_data = x[-1].detach().cpu().numpy()
    input_data = input_data.reshape(28,28)
    plt.imshow(input_data, cmap='gray', interpolation='none')
    plt.axis('off')
    IMG_NAME = PATH_OUTPUT + "example_input" + str(model.epoch) + ".png"
    logging.info(f"Saving image of last input to {IMG_NAME}")
    plt.savefig(IMG_NAME)

    # plot the last image currently saved in model
    ax = plt.figure(figsize=(3,3))
    output_data = model.rho[-1][-1].detach().cpu().numpy()
    output_data = output_data.reshape(28,28)
    plt.imshow(output_data, cmap='gray', interpolation='none')
    plt.axis('off')
    IMG_NAME = PATH_OUTPUT + "example_output" + str(model.epoch) + ".png"
    logging.info(f"Saving image of last output to {IMG_NAME}")
    plt.savefig(IMG_NAME)
    




if __name__ == '__main__':

    # path to folder of this file
    PATH_SCRIPT = os.path.dirname(os.path.realpath(__file__))

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
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, worker_init_fn=dataloader_seed_worker, drop_last=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, worker_init_fn=dataloader_seed_worker, drop_last=True)

    logging.info(f'Total training batches: {len(train_loader)}')
    logging.info(f'Total validation batches: {len(val_loader)}')
    logging.info(f'Total testing batches: {len(test_loader)}')

    if rec_weights:
        logging.info(f'Recording weights after every evaluation: {rec_weights}')
        weights_time_series = []
        bw_weights_time_series = []
    if rec_activations or rec_noise:
        logging.info(f'Recording during evaluation after every presentation time: Weights: {rec_weights}, Activations: {rec_activations}, Noise: {rec_noise}')


    # if a model file has been passed, load and do not train
    if args.load:
        with open(args.load, 'rb') as input:
            if use_cuda  == False:
                model = CPU_Unpickler(input).load()
                # we also need to set all device variables to cpu
                for layer in model.layers:
                    layer.device = 'cpu'
                model.device = 'cpu'
            else:
                model = pickle.load(input)
            logging.info(f"Loaded model {args.load}")

    else:

        # training

        if algorithm == 'PAL':
            model = MLPNet(batch_size, lr_multiplier, lr_factors, tau, dt, beta, algorithm, model_variant, target_type, presentation_steps, with_optimizer,
                           bw_lr_factors = bw_lr_factors, regularizer = regularizer, tau_xi = tau_xi, tau_HP = tau_HP, tau_LO = tau_LO, sigma = sigma, wn_sigma = wn_sigma)
        else:
            model = MLPNet(batch_size, lr_multiplier, lr_factors, tau, dt, beta, algorithm, model_variant, target_type, presentation_steps, with_optimizer, wn_sigma=wn_sigma)
        
        model.epoch = 0

        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # criterion = nn.CrossEntropyLoss()

        val_loss = []

        # create output directory if it doesn't exist
        if not(os.path.exists(PATH_OUTPUT)):
            logging.info(f"{PATH_OUTPUT} doesn't exists, creating")
            os.makedirs(PATH_OUTPUT)

        # save model at init
        with open(PATH_OUTPUT + 'MLPNet_epoch0.pkl', 'wb') as output:
                    pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
                    logging.info(f'Saved model to {output.name}')
        # evaluate model on test set
        logging.info("Evaluating model before training (val+test)")

        val, weights_arr, bw_weights_arr = validate_model(model, val_loader)
        val_loss.append(val)
        if weights_arr is not None:
            weights_time_series.append(weights_arr)
        if bw_weights_arr is not None:
            bw_weights_time_series.append(bw_weights_arr)   

        test_model(model, test_loader)

        logging.basicConfig(format='Train model -- %(levelname)s: %(message)s',
                    level=logging.INFO, force=True)

        logging.info('Starting training')

        # for epoch in tqdm(range(epochs), desc="Epochs"):
        for epoch in range(epochs):
            # training
            correct_cnt, summed_loss = 0, 0
            total_cnt = 0
            summed_loss = 0
            model.epoch += 1
            model.train()
            for batch_idx, (x, label) in enumerate(tqdm(train_loader, desc="Batches", disable=tqdm_disabled)):
            # for batch_idx, (x, label) in enumerate(train_loader):
                # we need to flatten for MNIST and MLPNet
                x = x.reshape([batch_size, 28 * 28])
                # we use an autoencoder
                # target = x
                # optimizer.zero_grad()
                if use_cuda:
                    # x, target = x.cuda(), target.cuda()
                    x = x.cuda()

                for update_i in range(presentation_steps):
                    model.update(x, x)

                loss = model.errors[-1]
                out = model.rho[-1]
                # loss = criterion(out, target)
                # _, pred_label = torch.max(out, 1)
                total_cnt += x.shape[0]
                # correct_cnt += (pred_label == torch.max(target, 1)[1]).sum()
                summed_loss += loss.detach().cpu().numpy()
                # loss.backward()
                # optimizer.step()
                if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                    logging.info(f'Epoch: {epoch+1}, batch index: {batch_idx + 1}, train loss: {(np.abs(summed_loss).sum(1)/total_cnt).mean(0):.9f}')

            # validate
            val, weights_arr, bw_weights_arr = validate_model(model, val_loader)
            val_loss.append(val)
            if weights_arr is not None:
                weights_time_series.append(weights_arr)
            if bw_weights_arr is not None:
                bw_weights_time_series.append(bw_weights_arr)   

            # after every epoch, save model

            with open(PATH_OUTPUT + 'MLPNet_epoch' + str(epoch+1) + '.pkl', 'wb') as output:
                pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
                logging.info(f'Saved model to {output.name}')

        with open(PATH_OUTPUT + "val_loss.pkl", "wb") as output:
            pickle.dump(val_loss, output)
            logging.info(f"Saving loss to {output.name}")

            # plot val loss
            ax = plt.figure(figsize=(7,5))
            plt.plot(val_loss)
            IMG_NAME = PATH_OUTPUT + "val_loss.png"
            logging.info(f"Saving plot of validation loss to {IMG_NAME}")
            plt.savefig(IMG_NAME)

        if rec_weights:

            # generate plot of angle between B and W.T
            deg_time_series = []

            for weights, bw_weights in zip(weights_time_series, bw_weights_time_series):
                # for every recorded time step, calculate cos_sim
                # leave out first entry of weights, as it is not used to transport errors
                deg_time_series.append([deg(cos_sim(W.T,B)) for W, B in zip(weights[1:], bw_weights[1:])])

            ax = plt.figure(figsize=(7,5))
            lines = plt.plot(deg_time_series)
            labels = ['layer ' + str(i+1) for i in range(len(model.layers))]
            plt.legend(lines, labels)
            plt.xlabel('Epochs')
            plt.ylabel('alignment [deg]')
            # plt.legend()
            IMG_NAME = PATH_OUTPUT + "deg_time_series.png"
            logging.info(f"Saving plot of angle between W.T and B to {IMG_NAME}")
            plt.savefig(IMG_NAME)

            # save weights
            with open(PATH_OUTPUT + "weights.pkl", "wb") as output:
                pickle.dump(weights_time_series, output)
                logging.info(f"Saving weights to {output.name}")
            with open(PATH_OUTPUT + "bw_weights.pkl", "wb") as output:
                pickle.dump(bw_weights_time_series, output)
                logging.info(f"Saving backwards weights to {output.name}")
            with open(PATH_OUTPUT + "deg_time_series.pkl", "wb") as output:
                pickle.dump(deg_time_series, output)
                logging.info(f"Saving angle between W.T and B to {output.name}")


    
    # evaluate model on test set
    test_model(model, test_loader)

