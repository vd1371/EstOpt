# Loading dependencies
import os
import sys
import time
import matplotlib.pyplot as plt
import pandas as pd
import pprint

from DataLoader import EstOptLoader
from Logger import Logger


import tensorflow as tf
import keras
from keras.models import Sequential, load_model
import keras.losses
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l1, l2
from keras.models import model_from_json
from sklearn.metrics import accuracy_score

class EstOpt:
	def __init__(self, **params):
		self.directory = "Reports/DNN/"
		self.log = Logger(address = f"{self.directory}Log.log")
		self.set_hyperparamters(**params)
		dl = EstOptLoader(**params)

		self.X_train, self.X_cv, self.X_test, \
			self.Y_train, self.Y_cv, self.Y_test, \
			self.indices_train, self.indices_cv, self.indices_test = dl.load_binaries()

		self.input_dim = len(self.X_train.columns)
		self.output_dim = len(self.Y_train.columns)

	def set_hyperparamters(self, **params):
		'''Setting hyperparameters'''
		self.split_size = params.pop("split_size", 0.3)
		self.should_shuffle = params.pop("should_shuffle", True)
		self.layers = params.pop("layers", [200])
		self.epochs = params.pop("epochs", 100)
		self.min_delta = params.pop("min_delta", 1)
		self.patience = params.pop("patience", 50)
		self.batch_size = params.pop("batch_size", 256)
		self.should_early_stop = params.pop("should_early_stop", False)
		self.regul_type = params.pop("regularization_type", "L2")
		self.reg_param = params.pop("reg_param", 0.000001)
		self.random_state = params.pop("random_state", 165)
		self.should_plot_live_error = params.pop("should_plot_live_error", True)
		self.should_checkpoint = params.pop("should_checkpoint", False)
		self.decimal = params.pop('decimal', True)
		self.final_activation_func = params.pop("final_activation_func", 'sigmoid')
		self.loss_func = params.pop("loss_func", 'binary_crossentropy')

		if self.regul_type.lower() == 'l2':
			self.l = l2
		else:
			self.l = l1

		self.input_activation_func = 'tanh'
		self.hidden_activation_func = 'relu'
		self.name = 'EstOpt'
		self.optimizer = 'RMSProp'

		self.log.info(pprint.pformat({'layers': self.layers,
									'input_activation_func': "tanh",
									'hidden_activation_func': "relu",
									'final_activation_func': self.final_activation_func,
									'loss_func': self.loss_func,
									'epochs': self.epochs,
									'min_delta': self.min_delta,
									'patience': self.patience,
									'batch_size': self.batch_size,
									'should_early_stop': self.should_early_stop,
									'regularization_type': self.regul_type,
									'reg_param': self.reg_param,
									'random_state': self.random_state}))

	def _get_call_backs(self):
		# Creating Early Stopping function and other callbacks

		#TODO: Complete

		call_back_list = []
		early_stopping = EarlyStopping(monitor='loss',
										min_delta = self.min_delta,
										patience=self.patience,
										verbose=1,
										mode='auto') 
		plot_losses = PlotLosses()
	
		if self.should_early_stop:
			call_back_list.append(early_stopping)
		if self.should_plot_live_error:
			call_back_list.append(plot_losses)
		if self.should_checkpoint:
			checkpoint = ModelCheckpoint(os.path.join(self.directory,
													f'{self.name}-BestModel.h5'),
													monitor='val_loss',
													verbose=1,
													save_best_only=True,
													mode='auto')
			call_back_list.append(checkpoint)

		return call_back_list

	def _construct_model(self, reg = None):

		model = Sequential()
		model.add(Dense(self.layers[0],
						input_dim = self.input_dim,
						activation = self.input_activation_func,
						kernel_regularizer=self.l(self.reg_param)))
		for ind in range(1,len(self.layers)):
			model.add(Dense(self.layers[ind],
							activation = self.hidden_activation_func,
							kernel_regularizer=self.l(self.reg_param)))
		model.add(Dense(self.output_dim, activation = self.final_activation_func))
		 
		# Compile model
		model.compile(loss=self.loss_func,
						optimizer=self.optimizer,
						metrics = ['mse'])

		return model

	def fit_model(self, warm_up = False):

		constructed = False
		if warm_up:
			try:
				self.load_model()
				constructed = True
				self.log.info("\n\n------------\nA trained model is loaded\n------------\n\n")
			except OSError:
				print ("The model is not trained before. No saved models found")

		if not constructed:
			# Creating the structure of the neural network
			self.model = self._construct_model()
			
			# A summary of the model
			stringlist = []
			self.model.summary(print_fn=lambda x: stringlist.append(x))
			short_model_summary = "\n".join(stringlist)
			self.log.info(short_model_summary)
		call_back_list = self._get_call_backs()

		start = time.time()
		# Fit the model
		hist = self.model.fit(self.X_train.values, self.Y_train.values,
							validation_data=(self.X_cv, self.Y_cv),
							epochs=self.epochs,
							batch_size=self.batch_size,
							verbose = 2, shuffle=True, callbacks=call_back_list)

		# Logging call_back history
		hist_df = pd.DataFrame.from_dict(hist.history)
		hist_df.to_csv(f"{self.directory}/{self.loss_func}-hist.csv")
		print (f"********* {time.time()-start:.4f} ***********")

		# Closing the plot losses
		try:
			for call_back in call_back_list:
				call_back.closePlot()
		except:
			pass
		
		# Evaluate the model
		train_scores = self.model.evaluate(self.X_train.values, self.Y_train.values, verbose=2)
		cv_scores = self.model.evaluate(self.X_cv, self.Y_cv, verbose=2)
		test_scores = self.model.evaluate(self.X_test, self.Y_test, verbose=2)
			
		print ()
		print (f'Trian_err: {train_scores}, Cv_err: {cv_scores}, Test_err: {test_scores}')
		self.log.info(f'Trian_err: {train_scores}, Cv_err: {cv_scores}, Test_err: {test_scores}')

		self.save_model()
		self.get_report()

	def get_report(self, load_model = False):

		if load_model:
			self.load_model()

		temp_df = self.Y_test.copy()

		y_test_pred = self.model.predict(self.X_test)
		cols = self.Y_test.columns
		pred_df = pd.DataFrame(y_test_pred, columns = cols)

		for col in self.Y_test.columns:
			temp_df[col + "-pred"] = pred_df[col]

		temp_df.to_csv("PredOnTest.csv")

	def load_model(self):
		
		# load json and create model
		model_type = 'BestModel' if self.should_checkpoint else 'SavedModel'
		self.model = load_model(self.directory + "/" +  f"{self.name}-{model_type}.h5")

	def save_model(self):
		save_address = self.directory + "/" + self.name 
		self.model.save(save_address + "-SavedModel.h5", save_format = 'h5')

	def logistic_regression(self):

		from sklearn.linear_model import LogisticRegression
		self.model = LogisticRegression()
		self.model.fit(self.X_train.values, self.Y_train.values.reshape(-1))

class PlotLosses(tf.keras.callbacks.Callback):
	
	def __init__(self, num = 0):
		self.num = num
	
	def on_train_begin(self, logs={}):
		self.i = 0
		self.x = []
		self.losses = []
		self.val_losses = []
		
		self.logs = []

	def on_epoch_end(self, epoch, logs={}):
		
		self.logs.append(logs)
		self.x.append(self.i)
		self.losses.append(logs.get('loss'))
		self.val_losses.append(logs.get('val_loss'))
		self.i += 1
		
		plt.ion()
		plt.clf()
		plt.title(f'Step {self.num}-epoch:{epoch}')
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		
		pointer = -100 if self.i > 100 else 0
		
		plt.plot(self.x[pointer:], self.losses[pointer:], label="loss")
		plt.plot(self.x[pointer:], self.val_losses[pointer:], label = 'cv_loss')
		
		plt.legend()
		plt.grid(True, which = 'both')
		plt.draw()
		plt.pause(0.000001)
	
	def closePlot(self):
		plt.close()



if __name__ == "__main__":

	# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
	myanalysis = EstOpt(split_size =  0.2,
						should_shuffle =  True,
						layers = [400, 400],
						epochs = 1000,
						min_delta = 1,
						patience = 50,
						batch_size = 1024,
						should_early_stop = False,
						regul_type = "l2",
						reg_param = 0.0000001,
						random_state = 165,
						should_plot_live_error = False,
						should_checkpoint = True,
						loss_func = 'mse',
						final_activation_func = 'sigmoid')
	
	myanalysis.fit_model(warm_up = False)
	# myanalysis.logistic_regression()
	myanalysis.get_report(load_model = False)