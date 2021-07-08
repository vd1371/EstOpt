'''
Estimation of optimization

This code intends to serve for the estimation of optimization project.
The idea is to use trained DNN models to find GA results.
Then, train another DNN on it to estimate GA results
'''
import os
import numpy as np
import pandas as pd
import pprint
import ast
from importlib import reload

from _utils import *


# from keras import backend as K
# os.environ['KERAS_BACKEND'] = 'theano'
# reload(K)
# from keras.models import load_model
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

RECON = 3
REHAB = 2
MAINT = 1
DONOT = 0

class GAOpt:

	def __init__(self):

		self.set_ga_chars()
		self.load_models()

		# To make code compatiable to the GIAMS codes
		self.n_elements = 3
		self.n_steps = 10
		self.dt = 2
		self.direc = "./results/"

	def load_models(self):

		import keras
		from keras import backend as K
		os.environ['KERAS_BACKEND'] = 'theano'
		reload(K)
		from keras.models import load_model

		"""Loading DNN files"""
		## TODO: Write this method
		self.user_cost_model = load_model("./models/User_df-BestModel.h5")
		self.agency_cost_model = load_model("./models/Agency_df-BestModel.h5")
		self.utility_model = load_model("./models/Utility_df-BestModel.h5")

	def create_bridge(self):

		prm = create_synthesized_assets()
		# prm = get_bridge_10_NBI()

		return prm

	def set_ga_chars(self, **params):

		"""Set the GA hyperparameters"""
		self.crossver_prob = params.pop("crossver_prob", 0.8)
		self.mutation_prob = params.pop("mutation_prob", 0.05)
		self.population_size = params.pop("population_size", 200)
		self.n_generations = params.pop("n_generations", 200)
		self.n_elites = params.pop("n_elites", 10)
		self.verbose = params.pop("verbose", False)
		# This is a maximization problem
		self.sorting_order = False
		self.patience = 50

		'''
		This array of probability will be used in the selection phase. The selection method is Ranking selection
		The worst: 1, second worst:2, .... Then they all will be devided by the sum so sum of probabilities be 1
		'''
		pop_size = self.population_size
		self.p = [(pop_size - i) / ((pop_size + 1)*pop_size/2) for i in range(pop_size)]

		
	def optimize(self, bridge = None):
		"""GA optimizaiton main loop"""
		taboo_list = []
		best_values = []
		first = True

		if bridge is None:
			bridge = self.create_bridge()

		for n_gener in range(self.n_generations):

			if first:
				gener_df = init_gener(bridge,
										taboo_list,
										check_policy_binary,
										**self.__dict__)
				first = False
			else:
				gener_df = next_gener(gener_df,
										taboo_list,
										check_policy_binary,
										**self.__dict__)

			gener_df = eval_gener(gener_df,
									check_policy_obj,
									**self.__dict__)

			# best_value = gener_df.iloc[0, -1]
			# best_values.append(best_value)

			if self.verbose:
				print ('-->> Generation', n_gener, best_value)

		return gener_df.iloc[0, :].values, bridge['id']

#-----------------------------------------------------------------------------------#
#
# The begining of executers 
#
#-----------------------------------------------------------------------------------#
def go_sane(N = 100, batch_size = 10):
	"""A function for parallel processing"""

	print ("batch_size", batch_size)

	gaopt_instance = GAOpt()
	cols = create_df_columns(gaopt_instance)

	holder, indices, start = [], [], time.time()
	batch_number = 1
	for i in range(1, N+1):

		results, id_ = gaopt_instance.optimize()

		holder.append(results)
		indices.append(id_)

		if i % batch_size == 0:
			holder, start, batch_number = save_batch(gaopt_instance.direc,
										holder,
										cols,
										indices,
										batch_number,
										start)

def go_indiana(dict_of_assets):
	"""A function for parallel processing"""

	gaopt_instance = GAOpt()
	cols = create_df_columns(gaopt_instance)

	holder, indices, start = [], [], time.time()
	batch_number = 1
	for idx, bridge_params in dict_of_assets.items():

		results, id_ = gaopt_instance.optimize(bridge_params)

		holder.append(results)
		indices.append(id_)

	holder, start, batch_number = save_batch(gaopt_instance.direc,
								holder,
								cols,
								indices,
								batch_number,
								start)
			
			


if __name__ == "__main__":
	import time

	dict_of_assets = input()
	dict_of_assets = ast.literal_eval(dict_of_assets)

	go_indiana(dict_of_assets)

	# start = time.time()
	# go_sane(N = 1, batch_size = 1)

	# print ("Done")
	# print (time.time()-start)