#Loading dependencies
import os
import sys
import time
import matplotlib.pyplot as plt
import pandas as pd
import pprint

from DataLoader import EstOptLoader
from ClassificationReport import evaluate_classification
from Logger import Logger

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from _Utils import save_model


class EstOpt:
	def __init__(self, **params):
		
		self.set_hyperparamters(**params)
		dl = EstOptLoader(**params)

		self.X_train, self.X_cv, self.X_test, \
			self.Y_train, self.Y_cv, self.Y_test, \
			self.indices_train, self.indices_cv, self.indices_test = dl.load_integers(n = None)

		self.input_dim = len(self.X_train.columns)
		self.output_dim = len(self.Y_train.columns)

	def set_hyperparamters(self, **params):
		
		self.n_estimators = params.pop('n_estimators', 100) 
		self.bootstrap = params.pop('bootstrap', True)
		self.max_depth = params.pop('max_depth', 40)
		self.max_features = params.pop('max_features', 2)
		self.min_samples_leaf = params.pop('min_samples_leaf', 4)
		self.min_samples_split = params.pop('min_samples_split', 0.6)
		self.n_jobs = params.pop('n_jobs', 1)
		self.verbose = params.pop('verbose', 1)
		self.random_state = params.pop('random_state', 165)


	def fit_model(self, model_name = 'RF', step = None):

		self.directory = f"Reports/{model_name}/"
		self.log = Logger(address = f"{self.directory}Log.log")

		for i in range (self.output_dim):

			if not step is None:
				i = step 

			start1 = time.time()

			Y_train, Y_test = self.Y_train.iloc[:, i], self.Y_test.iloc[:, i]

			if model_name == "RF":
				model = RandomForestClassifier(n_estimators=self.n_estimators, 
												max_depth=self.max_depth,
												min_samples_split=self.min_samples_split, 
												min_samples_leaf=self.min_samples_leaf,
												max_features=self.max_features,
												bootstrap=self.bootstrap,
												n_jobs=self.n_jobs,
												verbose=self.verbose,
												random_state = self.random_state)

			elif model_name == "DT":
				model = DecisionTreeClassifier(max_depth=self.max_depth,
												min_samples_split=self.min_samples_split, 
												min_samples_leaf=self.min_samples_leaf,
												random_state = self.random_state)
			elif model_name == "Logit":
				model = LogisticRegression(C = 100, fit_intercept = True, penalty= 'l2', solver = 'liblinear')


			model.fit(self.X_train.values, Y_train.values)
			evaluate_classification([f'OnTrain-{i}', self.X_train, Y_train, self.indices_train],
									[f'OnTest-{i}', self.X_test, Y_test, self.indices_test],
									direc = self.directory,
									model = model,
									model_name = f'{model_name}-{i}',
									logger = self.log,
									slicer = 1)

			save_model(model, self.directory, f"{model_name}-{i}")

			print ("------->", f"Step:{i} done in {time.time()-start1:.2f}")

			if not step is None:
				break

	
	def tune_trees(self):

		grid = {'n_estimators': [100, 200, 500, 800, 1000],
								'max_features': ['log2', 'sqrt', None],
								'max_depth': [10, 20, 50, 80, 100],
								'min_samples_leaf': [1, 10, 20, 50],
								'min_samples_split': [2, 20, 40, 100],
								'bootstrap': [True]}

		# grid = {'n_estimators': [100, 200],
		# 						'max_features': ['sqrt', None],
		# 						'max_depth': [10],
		# 						'min_samples_leaf': [1],
		# 						'min_samples_split': [2],
		# 						'bootstrap': [True]}

		model = RandomForestClassifier(random_state = self.random_state)
		search_models = GridSearchCV(
				estimator = model,
				param_grid = grid,
				cv = 5,
				verbose=2,
				n_jobs = self.n_jobs
				)

		search_models.fit(self.X_train, self.Y_train.iloc[:, 0])
		 
		self.log.info(f"\n\nBest params:\n{pprint.pformat(search_models.best_params_)}\n")
		self.log.info(f"\n\nBest score: {search_models.best_score_:0.4f}\n\n")
		print (search_models.best_score_)

if __name__ == "__main__":

	import time

	myanalysis = EstOpt(n_estimators = 500,
						max_depth = 100,
						max_features = 'auto',
						min_samples_split = 2,
						min_samples_leaf = 1,
						n_jobs = 3,
						split_size =  0.2,
						should_shuffle =  True,
						random_state = 165)

	myanalysis.fit_model(model_name = "RF", step = 6)
	# myanalysis.tune_trees()

	