import numpy as np
from microcircuit import *
import logging
import dill

logging.basicConfig(format='Train model -- %(levelname)s: %(message)s',
                    level=logging.INFO)


def save(MC_list, name='model', path=None):
	if path is None:
		with open(name + '.pkl', 'wb') as output:
			dill.dump(MC_list, output, dill.HIGHEST_PROTOCOL)
	else:
		with open(path + "/" + name + '.pkl', 'wb') as output:
			dill.dump(MC_list, output, dill.HIGHEST_PROTOCOL)
dill
def load(file='model.pkl'):
	with open(file, 'rb') as input:
		return dill.load(input)