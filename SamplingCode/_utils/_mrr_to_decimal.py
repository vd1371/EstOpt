import numpy as np
import pandas as pd

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