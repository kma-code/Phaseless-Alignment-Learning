import numpy as np
from microcircuit import *
import logging
import pickle

logging.basicConfig(format='Train model -- %(levelname)s: %(message)s',
                    level=logging.INFO)


def save(MC_list, name='model', path=None):
	if path is None:
		with open(name + '.pkl', 'wb') as output:
			pickle.dump(MC_list, output, pickle.HIGHEST_PROTOCOL)
	else:
		with open(path + "/" + name + '.pkl', 'wb') as output:
			pickle.dump(MC_list, output, pickle.HIGHEST_PROTOCOL)

def load(file='model.pkl'):
	with open(file, 'rb') as input:
		return pickle.load(input)