#Loading deoendencies
import pandas as pd
from sklearn.model_selection import train_test_split

from DataCleaning import base_directory

class EstOptLoader:

	def __init__(self, **params):
		'''Loading all files'''
		self.split_size = params.pop('split_size')
		self.should_shuffle = params.pop('should_shuffle')
		self.random_state = params.pop('random_state')

		self.df = pd.read_csv(f"{base_directory}//EstOptAll.csv", index_col = 0)
		print ('EstOptAll is loaded...')

	def _splitter(self, X, Y, indices):

		# Splitting to train and test
		X_train, X_temp, Y_train, Y_temp, \
			indices_train, indices_temp = train_test_split(X, Y, indices,
														test_size = self.split_size,
														shuffle = self.should_shuffle, 
														random_state = self.random_state)

		X_cv, X_test, Y_cv, Y_test, \
			indices_cv, indices_test = train_test_split(X_temp, Y_temp, indices_temp,
													test_size = 0.5,
													shuffle = self.should_shuffle,
													random_state = self.random_state)

		return X_train, X_cv, X_test, Y_train, Y_cv, Y_test, indices_train, indices_cv, indices_test
	
	def load_integers(self, n = None):

		cols = list(self.df.columns)
		# Let's find the index of the Eelem0-0 where the 
		# MRR plans start
		idx = cols.index("Eelem0-0")

		for n_element in range(3):
			for year in range(0, 20, 2):

				i = idx + n_element * 20 + year
				self.df[f'Eelem{n_element}At{year}'] = self.df.iloc[:, i] * 2 + self.df.iloc[:, i+1]

		self.df.drop(columns = cols[idx: idx + 60], inplace = True)

		if not n is None:
			self.df = self.df.iloc[:n, :]

		# Splitting X and Y
		idx = self.df.columns.get_loc("Eelem0At0")
		X, Y, indices = self.df.iloc[:, :idx], self.df.iloc[:, idx:], self.df.index

		return self._splitter(X, Y, indices)

	def load_binaries(self):

		cols = list(self.df.columns)
		# Let's find the index of the Eelem0-0 where the 
		# MRR plans start
		idx = cols.index("Eelem0-0")

		X, Y, indices = self.df.iloc[:, :idx], self.df.iloc[:, idx:], self.df.index
		return self._splitter(X, Y, indices)


		