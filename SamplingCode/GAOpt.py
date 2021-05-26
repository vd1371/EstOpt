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
from importlib import reload
from collections import Counter

import multiprocessing as mp
from multiprocessing import Process, Queue, Lock

from deap import tools

# from keras import backend as K
# os.environ['KERAS_BACKEND'] = 'theano'
# reload(K)
# from keras.models import load_model

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

RECON = 3
REHAB = 2
MAINT = 1
DONOT = 0

def inf_if_neative(x):
	return [np.inf] if x[0] < 0 else x


def mrr_to_decimal (mrr = None, n_elements = 3):

		if isinstance(mrr, np.ndarray):
			mrr = mrr.reshape((n_elements, -1))

		elif isinstance(mrr, pd.Series):
			mrr = mrr.values.reshape((n_elements, -1))

		elif isinstance(mrr, list):
			mrr = np.array(mrr).reshape((n_elements, -1))

		mrr_decoded = []
		for element_idx in range (n_elements):
			element_mrr = []
			for j in range(0, len(mrr[element_idx]), 2):
				element_mrr.append(int(str(int(mrr[element_idx][j]*10+ mrr[element_idx][j+1])), 2))
			mrr_decoded.append(element_mrr)

		return mrr_decoded

def check_policy(mrr):

	mrr_decimal = mrr_to_decimal(mrr)

	for elem_mrr in mrr_decimal:
		counts = Counter(elem_mrr)

		if counts[RECON] > 2:
			return False
		elif counts[REHAB] > 3:
			return False
		elif counts[MAINT] > 5:
			return False

	for i in range (len(mrr_decimal)):
		for val1, val2 in zip(mrr_decimal[i][:-1], mrr_decimal[i][1:]):
			if val1 * val2 > 0:
				return False

	return True

def check_policy2(vec):

	obj_value = vec[-1]
	mrr_decimal = mrr_to_decimal(vec[:-1])

	for elem_mrr in mrr_decimal:
		counts = Counter(elem_mrr)

		if counts[RECON] > 2 or counts[REHAB] > 3 or counts[MAINT] > 5:
			return -1000

	for i in range (len(mrr_decimal)):
		for val1, val2 in zip(mrr_decimal[i][:-1], mrr_decimal[i][1:]):
			if val1 * val2 > 0:
				return -1000

	return obj_value

class GAOpt:

	def __init__(self):

		self.set_ga_chars()
		self.load_models()

		self.first_bridge = True

	def _add_to_taboo_list(self, solut):
		self.taboo_list.append(hash(solut.tostring()))

	def _is_in_taboo_list(self, solut):
		return hash(solut.tostring()) in self.taboo_list

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

	def set_ga_chars(self, 
		crossver_prob = 0.8,
		mutation_prob = 0.05,
		population_size = 200,
		n_generations = 200,
		n_elites = 10,
		verbose = False):

		"""Set the GA hyperparameters"""
		self.crossver_prob = crossver_prob
		self.mutation_prob = mutation_prob
		self.population_size = population_size
		self.n_generations = n_generations
		self.n_elites = n_elites
		self.verbose = verbose
		# This is a maximization problem
		self.sorting_order = False
		self.patience = 50

		'''
		This array of probability will be used in the selection phase. The selection method is Ranking selection
		The worst: 1, second worst:2, .... Then they all will be devided by the sum so sum of probabilities be 1
		'''
		self.p = [(population_size - i) / ((population_size + 1)*population_size/2) for i in range(population_size)]

	def create_bridge(self):
		"""
		Create a random brige

		This method should:
			1) create a random bridge
			2) one-hot encode the results
		pass the created bridge results
		"""
		# To make code compatiable to the GIAMS codes
		self.n_elements = 3
		self.n_steps = 10
		self.dt = 2

		prm = {'id' : hash(str(np.random.random()*np.random.random())), 
				'length' : (np.random.random_integers(5, 1800) - 5) / 1795,
				'width' : (np.random.random_integers(3, 60) - 3) / 57,
				'vertical_clearance': (np.random.uniform(4, 7)-4) / 3,
				'ADT': (np.random.random_integers(100, 400000) - 100) / 399900,
				'truck_percentage': np.random.uniform(0, 0.5) / 0.5,
				'detour_length': (np.random.random_integers(1, 100) - 1) / 99,
				'skew_angle': np.random.random_integers(0, 45) / 45,
				'n_spans': (int(np.random.random_integers(1, 60)) - 1) / 59,
				'maint_duration': (np.random.random_integers(10, 60) - 10) / 50,
				'rehab_duration': (np.random.random_integers(120, 240) - 120) / 120,
				'recon_duration': (np.random.random_integers(300, 540) - 300) / 240,
				'speed_before': (np.random.random_integers(40, 90) - 40) / 50,
				'speed_after': (np.random.random_integers(15, 35) - 15) / 20,
				'drift': (np.random.uniform(0.01, 0.1) - 0.01) / 0.09,
				'volatility': (np.random.uniform(0.01, 0.1) - 0.01) / 0.09,
				'detour_usage_percentage': np.random.uniform(0, 0.99) / 0.99,
				'occurrence_rate': (np.random.uniform(0.001, 0.1) - 0.001) / 0.099,
				'dist_first_param': (np.random.uniform(3, 5) - 3) / 2,
				'dist_second_param': (np.random.uniform(0.01, 2) - 0.01) / 1.99,
				'deck_cond': (np.random.choice([9, 8, 7, 6, 5, 4]) - 4) / 5,
				'deck_age': (np.random.random_integers(1, 90) - 1) / 89,
				'superstructure_cond': (np.random.choice([9, 8, 7, 6, 5, 4]) - 4) / 5,
				'superstructure_age': (np.random.random_integers(1, 90) - 1) / 89,
				'substructure_cond': (np.random.choice([9, 8, 7, 6, 5, 4]) - 4) / 5,
				'substructure_age': (np.random.random_integers(1, 90) - 1) / 89
				}

		# Other columns are one-hot encoded previously so we need to 
		# do it manually
		categorical_info_dict = {
			'material': [1, 2, 3, 4, 5, 6],
			'design' : [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 14, 16, 21, 22],
			'hazus_class' : [f'HWB{val}' for val in [1, 10, 12, 15, 17, 22, 3, 5, 8,]],
			'road_class' : ['Local', 'Major', 'Minor', 'NHS'],
			'site_class' : ['A', 'B', 'C'],
			'deck_material' : [1, 2, 3, 8]
			}

		# Adding encoded parameters
		for key, ls in categorical_info_dict.items():
			chosen_one = np.random.choice(ls)
			for i, item in enumerate(ls[1:]):
				prm[f'{key}_{item}'] = 1 if item == chosen_one else 0

		# -----------------------------------------------------------------------------------#
		# To reporoduce the example in the GIAMS paper
		# -----------------------------------------------------------------------------------#

		# prm = {'id' : hash(str(np.random.random()*np.random.random())), 
		# 		'length' : (54.3 - 5) / 1795,
		# 		'width' : (16.8 - 3) / 57,
		# 		'vertical_clearance': (7-4) / 3,
		# 		'ADT': (12797 - 100) / 399900,
		# 		'truck_percentage': 0.05 / 0.5,
		# 		'detour_length': (6 - 1) / 99,
		# 		'skew_angle': 6 / 45,
		# 		'n_spans': (3 - 1) / 59,
		# 		'maint_duration': (30 - 10) / 50,
		# 		'rehab_duration': (180 - 120) / 120,
		# 		'recon_duration': (360 - 300) / 240,
		# 		'speed_before': (60 - 40) / 50,
		# 		'speed_after': (30 - 15) / 20,
		# 		'drift': (0.1 - 0.01) / 0.09,
		# 		'volatility': (0.01 - 0.01) / 0.09,
		# 		'detour_usage_percentage': 0.1 / 0.99,
		# 		'occurrence_rate': (0.3 - 0.001) / 0.099,
		# 		'dist_first_param': (2.1739 - 3) / 2,
		# 		'dist_second_param': (4 - 0.01) / 1.99,
		# 		'deck_cond': (5 - 4) / 5,
		# 		'deck_age': (14 - 1) / 89,
		# 		'superstructure_cond': (7 - 4) / 5,
		# 		'superstructure_age': (14 - 1) / 89,
		# 		'substructure_cond': (7 - 4) / 5,
		# 		'substructure_age': (14 - 1) / 89
		# 		}

		# categorical_info_dict = {
		# 	'material': [1, 2, 3, 4, 5, 6],
		# 	'design' : [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 14, 16, 21, 22],
		# 	'hazus_class' : [f'HWB{val}' for val in [1, 10, 12, 15, 17, 22, 3, 5, 8,]],
		# 	'road_class' : ['Local', 'Major', 'Minor', 'NHS'],
		# 	'site_class' : ['A', 'B', 'C'],
		# 	'deck_material' : [1, 2, 3, 8]
		# 	}

		# # Adding encoded parameters
		# for key, ls in categorical_info_dict.items():
		# 	chosen_one = np.random.choice(ls)
		# 	for i, item in enumerate(ls[1:]):
		# 		prm[f'{key}_{item}'] = 0

		# prm['material_4'] = 1
		# prm['design_2'] = 1
		# prm['hazus_class_HWB15'] = 1
		# prm['road_class_Major'] = 1
		# prm['site_class_B'] = 1
		### Since the deck material is 1, we don't set it as 1
		### prm['deck_material'] = 1

		if self.verbose:
			pprint.pprint(prm)

		return prm

	def eval_gener(self, df):

		def get_cols(df, ls):
			output = []
			for item in ls:
				for col in df.columns:
					if item in col: output.append(col)
			return output

		##TODO: Check the column order and validate with the source files

		start = time.time()
		for _ in range(100):

			user_costs = self.user_cost_model(df.drop(columns = get_cols(df, ['UserCost', 'AgencyCost', 'Utility', 'Obj',
															'width', 'vertical_clearance', 'design_'])), training = False).numpy()

			# agency_costs = self.agency_cost_model(df.drop(columns = get_cols(df, ['UserCost', 'AgencyCost', 'Utility', 'Obj',
			# 												'ADT', 'truck_percentage', 'detour_length',
			# 												'_duration', 'speed_', 'drift',
			# 												'volatility', 'detour_usage_percentage'])), training = False).numpy()

			utilities = self.utility_model(df.drop(columns = get_cols(df, ['UserCost', 'AgencyCost', 'Utility', 'Obj',
														'length', 'width', 'vertical_clearance',
														'design_', 'ADT', 'truck_percentage',
														'_duration', 'speed_', 'drift',
														'volatility', 'detour_usage_percentage'])), training = False).numpy()

		print (time.time()-start)
		input()

		# Finding the objective function (It is currently based on GIAMS example 1)
		user_costs = user_costs.reshape(-1)
		user_costs[user_costs < 0] = np.inf
		user_costs = user_costs.reshape(-1, 1)


		df['Obj'] = (utilities / user_costs** 0.2)

		# Checking the values
		df['Obj'] = df.loc[:, 'Eelem0-0': 'Obj'].apply(check_policy2, axis = 1)

		# Sorting values
		df.sort_values('Obj', inplace = True, ascending = self.sorting_order)

		return df

	def init_gener(self, bridge = None):
		"""Initialzing the GA first generation"""
		if bridge is None:
			bridge = self.create_bridge()

		holder = []
		bridge_vals = list(bridge.values())
		for _ in range(self.population_size):

			found = False
			while not found:
				p = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
				random_mrr = np.random.choice([0, 1], size = self.n_elements * self.n_steps*self.dt, p = [1-p, p])
				if np.sum(random_mrr) > 3 and \
					np.sum(random_mrr) < 34 and \
						check_policy(random_mrr) and \
							not self._is_in_taboo_list(random_mrr):

					found = True
					self._add_to_taboo_list(random_mrr)
					holder.append(bridge_vals + random_mrr.tolist())

			# holder.append(bridge_vals + [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
			# 							0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
			# 							1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1])

		cols = list(bridge.keys())
		# Three elements were used in the original project
		for elem in range(3):
			# 20 years, biennially, binary representation --> 20 variables
			for step in range(20):
				cols.append(f'Eelem{elem}-{step}')

		# Converint to pandas dataframe
		df = pd.DataFrame(holder, columns = cols)

		# We need to update the id because we don't want the id
		# be the same for all individuals
		df['id'] = [hash(str(np.random.random()*np.random.random())) for _ in range (self.population_size)]
		df.set_index('id', inplace = True)

		return df

	def mate(self, chrom1, chrom2):

		# Two point crossover
		if np.random.random() < self.crossver_prob:
			chrom1, chrom2 = tools.cxTwoPoint(chrom1, chrom2)

		# Flipbit mutation
		chrom1 = tools.mutFlipBit(chrom1, self.mutation_prob)
		chrom2 = tools.mutFlipBit(chrom2, self.mutation_prob)

		return np.array(chrom1[0]), np.array(chrom2[0])

	def next_gener(self, df):
		
		next_gener_mrr = []

		# Elisitsm
		next_gener_mrr = df.loc[df.index[:self.n_elites], 'Eelem0-0': 'Eelem2-19'].values.tolist()

		# Choosing the ones with obb > 0
		valid_inds = df[df['Obj'] > 0]
		pop_size = len(valid_inds)
		probs = [(pop_size - i) / ((pop_size + 1)*pop_size/2) for i in range(pop_size)]

		while len(next_gener_mrr) < self.population_size:

			# Selecting parents
			parents_indices = np.random.choice(valid_inds.index, size = (2,), p= probs, replace = False)

			parent1, parent2 = df.loc[parents_indices, 'Eelem0-0': 'Eelem2-19'].values.tolist()

			offspring1, offspring2 = self.mate(parent1, parent2)

			for offspring in [offspring1, offspring2]:

				if check_policy(offspring) and not self._is_in_taboo_list(offspring):
					next_gener_mrr.append(offspring)
					self._add_to_taboo_list(offspring)

		df.loc[:, 'Eelem0-0': 'Eelem2-19'] = next_gener_mrr[:self.population_size]


		return df
		
	def optimize(self):
		"""GA optimizaiton main loop"""
		self.taboo_list = []
		best_values = []
		first = True
		for n_gener in range(self.n_generations):

			if first:
				# Creating the first generation
				gener_df = self.init_gener()
				first = False
			else:
				# Creating the new generation
				gener_df = self.next_gener(gener_df)

			# Evaluate the generation
			gener_df = self.eval_gener(gener_df)

			# best_value = gener_df.iloc[0, -1]
			# best_values.append(best_value)

			if self.verbose:
				print ('-->> Generation', n_gener, best_value)

		return gener_df.iloc[0, :].values


#-----------------------------------------------------------------------------------#
#
#
#
# The begining of executers 
#
#
#
#-----------------------------------------------------------------------------------#
def go_sane(N = 100, batch_size = 10):
	"""A function for parallel processing"""

	print ("batch_size", batch_size)

	gaopt_instance = GAOpt()

	cols = list(gaopt_instance.create_bridge().keys())[1:]
	# Three elements were used in the original project
	for elem in range(3):
		# 20 years, biennially, binary representation --> 20 variables
		for step in range(20):
			cols.append(f'Eelem{elem}-{step}')
	cols.append("obj")

	holder = []
	batch_number = 1
	start = time.time()
	for i in range(1, N+1):
		holder.append(gaopt_instance.optimize())

		if i % batch_size == 0:
			
			df = pd.DataFrame(holder, columns = cols)

			ID = str(time.time())[-7:]
			df.to_csv(f"./results/EstOpt-{batch_number}-{ID}.csv")

			print (f'Batch number {batch_number} is done in {(time.time()-start)/60:.2f} minutes')
			start = time.time()
			batch_number += 1
			holder = []
#-----------------------------------------------------------------------------------#
#
#
#
# The begining of parallel processors
#
#
#
#-----------------------------------------------------------------------------------#
def go_crazy(N = 10, q_out = None):
	"""A function for parallel processing"""
	gaopt_instance = GAOpt()
	for _ in range(N):
		q_out.put(gaopt_instance.optimize())

def GAOpt_parallel(N = 1000, batch_size = 100):

	results_queue = Queue()
	n_cores = mp.cpu_count()-2
	N_for_each_core = int(N/n_cores)
	mylock = Lock()

	# Creating and filling the pool
	pool = []
	for j in range (n_cores):
		worker = Process(target = go_crazy, args = (N_for_each_core, results_queue, ))
		pool.append(worker)

	print('starting processes...')
	for worker in pool:
		worker.start()

	all_samples = []
	done_workers = 0
	batch_number = 0

	cols = []
	# cols = list(GAOpt().create_bridge().keys())[1:]
	# Three elements were used in the original project
	for elem in range(3):
		# 20 years, biennially, binary representation --> 20 variables
		for step in range(20):
			cols.append(f'Eelem{elem}-{step}')

	start = time.time()
	while any(worker.is_alive() for worker in pool):

		while not results_queue.empty():
			sample = results_queue.get()

			if not sample is None:
				all_samples.append(sample)

		# Saving each batch
		if len(all_samples) > batch_size:
			batch_number += 1
			df = pd.DataFrame(all_samples[:batch_size], columns = cols)
			
			print (f'Batch number {batch_number} is done in {time.time()-start:.2f}')
			start = time.time()


			df.to_csv(f"./results/EstOpt-{batch_number}.csv")
			all_samples = all_samples[batch_size:]

	# Saving the last batch
	if len(all_samples) > 0:
		batch_number += 1
		df = pd.DataFrame(all_samples, columns = cols)
		df.to_csv(f"./results/EstOpt-{batch_number}.csv")
		print (f'Batch number {batch_number} is done')

	print('waiting for workers to join...')
	for worker in pool:
		worker.join()
	print('all workers are joined.\n')


if __name__ == "__main__":
	import time

	start = time.time()
	# myGAOpt = GAOpt()
	# mrr = myGAOpt.optimize()[-60:]

	# GAOpt_parallel(N = 1000000, batch_size = 10)

	go_sane(N = 1, batch_size = 1)

	print ("Done")
	print (time.time()-start)