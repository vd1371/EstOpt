import numpy as np
import pandas as pd

from ._add_to_taboo_list import _add_to_taboo_list
from ._is_in_taboo_list import _is_in_taboo_list

def init_gener(bridge, taboo_list,check_policy_binary, **params):
	"""Initialzing the GA first generation"""

	population_size = params.pop("population_size")
	n_elements = params.pop("n_elements")
	n_steps = params.pop("n_steps")
	dt = params.pop("dt")

	holder = []
	bridge_vals = list(bridge.values())

	for _ in range(population_size):

		found = False
		while not found:
			p = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
			random_mrr = np.random.choice([0, 1], size = n_elements * n_steps* dt, p = [1-p, p])
			if np.sum(random_mrr) > 3 and \
				np.sum(random_mrr) < 34 and \
					check_policy_binary(random_mrr) and \
						not _is_in_taboo_list(taboo_list, random_mrr):

				found = True
				_add_to_taboo_list(taboo_list, random_mrr)
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

	# Convert to pandas dataframe
	df = pd.DataFrame(holder, columns = cols)

	# We need to update the id because we don't want the id
	# be the same for all individuals
	df['id'] = [hash(str(np.random.random()*np.random.random())) for _ in range (population_size)]
	df.set_index('id', inplace = True)

	return df