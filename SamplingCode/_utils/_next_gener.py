import numpy as np

from ._add_to_taboo_list import _add_to_taboo_list
from ._is_in_taboo_list import _is_in_taboo_list
from ._mate import mate

def next_gener(df, taboo_list, check_policy_binary, **params):

	n_elites = params.pop("n_elites")
	population_size = params.pop("population_size")

	next_gener_mrr = []

	# Elisitsm
	next_gener_mrr = df.loc[df.index[:n_elites], 'Eelem0-0': 'Eelem2-19'].values.tolist()

	# Choosing the ones with obb > 0
	valid_inds = df[df['Obj'] > 0]
	pop_size = len(valid_inds)
	probs = [(pop_size - i) / ((pop_size + 1)*pop_size/2) for i in range(pop_size)]

	while len(next_gener_mrr) < population_size:

		# Selecting parents
		parents_indices = np.random.choice(valid_inds.index, size = (2,), p= probs, replace = False)

		parent1, parent2 = df.loc[parents_indices, 'Eelem0-0': 'Eelem2-19'].values.tolist()

		offspring1, offspring2 = mate(parent1, parent2, **params)

		for offspring in [offspring1, offspring2]:

			if check_policy_binary(offspring) and not _is_in_taboo_list(taboo_list, offspring):
				next_gener_mrr.append(offspring)
				_add_to_taboo_list(taboo_list, offspring)

	df.loc[:, 'Eelem0-0': 'Eelem2-19'] = next_gener_mrr[:population_size]

	return df