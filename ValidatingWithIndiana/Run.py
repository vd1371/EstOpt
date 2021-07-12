import numpy as np

from DataLoader import load_indiana_data
from ModelLoader import load_trained_model
from ClassificationReporter import get_classification_report
from ClassificationReporter import log_final_analysis

from Logger import Logger

def exec():

	data = load_indiana_data()
	logger = Logger(address = "Log.Log")

	n_steps = 10
	n_elements = 2

	holder = []
	for ne in range(n_elements):
		for step in range(n_steps):

			model = load_trained_model(step = step, ne = ne)

			classif_results = get_classification_report(data, model, step,
														ne,	logger)

		holder.append(classif_results)

	log_final_analysis(holder, logger)

	print ("Done")

if __name__ == "__main__":
	exec()



