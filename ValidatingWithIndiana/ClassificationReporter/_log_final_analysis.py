import numpy as np
import pprint

def log_final_analysis(holder, logger):
	all_results = {}
	for metric in holder[0].keys():
		tmp = []
		for item in holder:
			tmp.append(item[metric])

		all_results[metric] = np.mean(tmp)

	logger.info(pprint.pformat(all_results))

