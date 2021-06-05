#import random
#random.seed(786)
#import numpy as np
#np.random.seed(786)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.backend import epsilon
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.python.ops import math_ops
import pandas as pd
from utils import *
from sklearn.datasets import *
from sklearn.svm import SVC
from inf_density import InformationDensity
from egl import expected_gradient
from sklearn.metrics import precision_score
from load_model import create_model

# Weighted version of log_loss; allows us to weigh precision over recall.
# Used to train SSL model.
def log_loss(y_true, y_pred, weight=100):
	y_true = tf.convert_to_tensor(y_true)
	y_pred = tf.convert_to_tensor(y_pred)
	y_true = tf.cast(y_true, y_pred.dtype)
	epsilon_ = tf.constant(epsilon(), dtype=y_pred.dtype.base_dtype)
	y_pred = tf.clip_by_value(y_pred, epsilon_, 1. - epsilon_)
	bce = y_true * tf.math.log(y_pred + epsilon())
	bce += weight * (1 - y_true) * tf.math.log(1 - y_pred + epsilon())
	return -bce

# Weighted version of MSE; allows us to weigh precision over recall.
# Used to train AL model.
def mse(y_true, y_pred, tau=85):
	y_true = tf.convert_to_tensor(y_true)
	y_pred = tf.convert_to_tensor(y_pred)
	y_true = tf.cast(y_true, y_pred.dtype)
	mse_val = K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1)
	scaling = tf.math.exp((y_true - 1)/(2 * tau**2))
	return mse_val*scaling

class DeepLearningActiveLearning():
	def __init__(self,
				 scale = True,
				 load_model = True,
				 retrain_al = True,
				 retrain_ssl = True,
				 save_model = True,
				 model_str="my_model",
				 al_model_epochs = 1000,
				 ssl_model_epochs = 1000):
		self.load_model = load_model
		self.save_model = save_model
		self.model_str = model_str
		self.retrain_al = retrain_al
		self.retrain_ssl = retrain_ssl
		self.al_model_epochs = al_model_epochs
		self.al_model_loaded = False
		self.ssl_model_epochs = ssl_model_epochs
		self.ssl_model_loaded = False
		self.scale = scale

		self.initAL()
		self.initSSL()

	def compile_ssl_model(self):
		return create_model(_model = "VPLNN")

	def compile_al_model(self):
		return create_model(_model = "VQNN")

	def initAL(self):
		load_str = self.model_str + "_al"
		self.modelAL = self.compile_al_model()
		self.modelALScaler = StandardScaler()

		if self.load_model:
			try:
				self.modelAL.load_weights(f"{load_str}.hdf5")
				if self.scale:
					self.modelALScaler = pickle.load(open(load_str + '_scaler.pkl', 'rb'))
				self.al_model_loaded = True
				return
			except Exception as e:
				print(e)
				pass

	def initSSL(self):
		load_str = self.model_str + "_ssl"
		self.modelSSL = self.compile_ssl_model()
		self.modelSSLScaler = StandardScaler()

		if self.load_model:
			try:
				self.modelSSL.load_weights(f"{load_str}.hdf5")
				if self.scale:
					self.modelSSLScaler = pickle.load(open(load_str + '_scaler.pkl', 'rb'))
				self.ssl_model_loaded = True
				return
			except Exception as e:
				print(e)
				pass

	#We assume dataset possesses only training data
	def trainAL(self, dataset):
		if (not self.retrain_al) and self.al_model_loaded:
			#We successfully loaded the model and we do not wish to retrain it; return now.
			return

		print(f"Active Learning Retrain: {self.retrain_al} Loaded: {self.al_model_loaded}")

		#Load dataset; careful about size (> 3GB). May need to switch to dask from pandas.
		dataframe = pd.read_csv(dataset)
		#Randomly shuffle all data values
		dataframe = dataframe.sample(frac=1)

		#For active learning, we wish to predict the performance rank; drop from X.
		X = dataframe.drop("PerfRank", axis=1)
		y = dataframe["PerfRank"]

		save_str = self.model_str + "_al"

		if self.scale:
			if not self.al_model_loaded:
				self.modelALScaler.fit(X)
			if self.save_model:
				pickle.dump(self.modelALScaler, open(save_str + '_scaler.pkl', 'wb'))
			X = self.modelALScaler.transform(X)

		#Train AL model
		if self.save_model:
			#Add Model Checkpoint system
			model_checkpoint_callback = ModelCheckpoint(
											filepath=f"{save_str}.hdf5",
											save_weights_only=True,
											monitor='loss',
											mode='min',
											save_best_only=True)

			self.modelAL.fit(X, y, epochs=self.al_model_epochs, batch_size=4096, shuffle=True, callbacks=[model_checkpoint_callback])
		else:
			#Fit without checkpoints
			self.modelAL.fit(X, y, epochs=self.al_model_epochs, batch_size=4096, shuffle=True)

	#We assume dataset possesses only training data
	def trainSSL(self, dataset):
		if (not self.retrain_ssl) and self.ssl_model_loaded:
			#We successfully loaded the model and we do not wish to retrain it; return now.
			return

		print(f"Semi Supervised Learning Retrain: {self.retrain_al} Loaded: {self.ssl_model_loaded}")

		#Load dataset; careful about size (> 3GB). May need to switch to dask from pandas.
		dataframe = pd.read_csv(dataset)
		#Randomly shuffle all data values
		dataframe = dataframe.sample(frac=1)

		#For semi-supervised learning, we wish to predict the CorrectPrediction; drop from X.
		X = dataframe.drop("CorrectPrediction", axis=1)
		try:
			X = X.drop("PerfRank", axis=1)
		except:
			pass
		y = dataframe["CorrectPrediction"]

		save_str = self.model_str + "_ssl"

		if self.scale:
			if not self.ssl_model_loaded:
				self.modelSSLScaler.fit(X)
			if self.save_model:
				pickle.dump(self.modelSSLScaler, open(save_str + '_scaler.pkl', 'wb'))
			X = self.modelSSLScaler.transform(X)

		#Train SSL model
		if self.save_model:
			#Add Model Checkpoint system
			model_checkpoint_callback = ModelCheckpoint(
											filepath=f"{save_str}.hdf5",
											save_weights_only=True,
											monitor='loss',
											mode='min',
											save_best_only=True)

			self.modelSSL.fit(X, y, epochs=self.ssl_model_epochs, batch_size=16384, shuffle=True, callbacks=[model_checkpoint_callback])
		else:
			#Fit without checkpoints
			self.modelSSL.fit(X, y, epochs=self.ssl_model_epochs, batch_size=16384, shuffle=True)

	def get_ranks(self, HeuristicMatrix):
		if self.scale:
			HeuristicMatrix = self.modelALScaler.transform(HeuristicMatrix)
			return self.modelAL.predict(HeuristicMatrix)

	#Given the Heuristic Matrix, query
	def query_index(self, HeuristicMatrix):
		return np.argmax(self.get_ranks(HeuristicMatrix))

	def get_pseudo_label(self, HeuristicMatrix):
		if self.scale:
			HeuristicMatrix = self.modelSSLScaler.transform(HeuristicMatrix)
			return np.squeeze(self.modelSSL.predict(HeuristicMatrix) > 0.5)		

	#Given the Heuristic Matrix, return boolean list of true values.
	def pseudo_label_index(self, HeuristicMatrix):
		if self.scale:
			HeuristicMatrix = self.modelSSLScaler.transform(HeuristicMatrix)
			return np.squeeze(self.modelSSL.predict(HeuristicMatrix) > 0.9)

def active_learn(model, X_train, y_train, X_test, y_test, use_ssl=True, dataset_name="Digits"):
	dlal = DeepLearningActiveLearning()

	Cid = InformationDensity()
	Cid.fit(X_train)

	#This represents the maximum score we can hope to achieve with the given model.
	model.fit(X_train, y_train)
	max_score = model.score(X_test, y_test)

	print(f"Best Score: {max_score} | Num Training Points: {len(X_train)}")

	X_unlabel, X_label, y_unlabel, y_label = init_split(X_train, y_train, randomSplit=False, add_medoids=True)
	NumPointsLst = [len(X_label)]

	X_fit, y_fit = X_label, y_label

	model.fit(X_fit, y_fit)
	CurAccuracy = [model.score(X_test, y_test)]

	print(f"Number of Labeled Points: {NumPointsLst[-1]} Score: {CurAccuracy[-1]}")

	inc_pseudos = 0
	num_pseudos = 0

	while CurAccuracy[-1] < max_score:
		#Query and Label Vectors
		HeuristicRankMatrix = accumulate_heuristics(model, X_fit, y_fit, X_unlabel, y_unlabel, Cid, rank_conv = True)

		indices = [dlal.query_index(HeuristicRankMatrix)]

		#Compare against true values (Sanity Check / Curiosity - not reported values.)
		"""
		perfLst = perf_inc_rank(model, X_fit, y_fit, X_unlabel, y_unlabel, X_test, y_test)
		perfPredLst = dlal.get_ranks(HeuristicRankMatrix)
		print(f"MSE: {(np.square(perfLst - perfPredLst)).mean()} | Optimal Chosen: {np.argmax(perfLst) == np.argmax(perfPredLst)}")
		"""

		X_unlabel, y_unlabel, X_label, y_label = migrate_vectors(X_unlabel, y_unlabel, X_label, y_label, indices)

		model.fit(X_label, y_label)

		if use_ssl:

			y_unlabel_pred = model.predict(X_unlabel)


			#Execute SSL loop
			HeuristicMatrix = accumulate_heuristics(model, X_label, y_label, X_unlabel, y_unlabel, Cid, rank_conv = False)
			pseudo_bool_lst = dlal.pseudo_label_index(HeuristicMatrix)

			#Evaluate how well VPLNN does at identifying vectors
			"""
			pred_bool_lst = dlal.get_pseudo_label(HeuristicMatrix).astype(int)
			test_bool_lst = (y_unlabel_pred == y_unlabel).astype(int)
			print(f"PseudoLabel Precision: {precision_score(test_bool_lst, pred_bool_lst)}")
			print(f"Confident PseudoLabel Precision: {precision_score(test_bool_lst, pseudo_bool_lst.astype(int))}")
			"""

			#Enforce smoothness constraint
			pseudo_bool_lst = enforce_smoothness(model, X_unlabel, y_unlabel, X_label, y_label, pseudo_bool_lst)

			ssl_loop_counter = 0
			prev_bool_lst = None

			while (prev_bool_lst is None) or ((not np.array_equal(pseudo_bool_lst, prev_bool_lst)) and ssl_loop_counter < 5):
				#Ensure we don't run through the loop for too long without fresh data
				ssl_loop_counter += 1

				prev_bool_lst = pseudo_bool_lst

				X_fit, y_fit = merge_sets(X_unlabel[pseudo_bool_lst], y_unlabel_pred[pseudo_bool_lst], X_label, y_label)
		
				model.fit(X_fit, y_fit)
				y_unlabel_pred = model.predict(X_unlabel)
				print(f"Number of pseudo labels (Loop {ssl_loop_counter}): {len(X_fit) - len(X_label)}")
				
				HeuristicMatrix = accumulate_heuristics(model, X_fit, y_fit, X_unlabel, y_unlabel_pred, Cid, rank_conv = False)
				pseudo_bool_lst = dlal.pseudo_label_index(HeuristicMatrix)
				#Enforce smoothness constraint
				pseudo_bool_lst = enforce_smoothness(model, X_unlabel, y_unlabel_pred, X_fit, y_fit, pseudo_bool_lst)

			inc_pseudos += np.count_nonzero(np.not_equal(y_unlabel[pseudo_bool_lst], y_unlabel_pred[pseudo_bool_lst]))
			num_pseudos += np.count_nonzero(pseudo_bool_lst)

			indices = list(np.nonzero(pseudo_bool_lst))
			X_unlabel, y_unlabel_pred, X_label, y_label = migrate_vectors(X_unlabel, y_unlabel_pred, X_label, y_label, indices)

			for index in indices:
				y_unlabel = np.delete(y_unlabel, index)

			model.fit(X_label, y_label)
			y_unlabel_pred = model.predict(X_unlabel)

		#Perform Analysis
		NumPointsLst.append(NumPointsLst[-1] + 1)
		CurAccuracy.append(model.score(X_test, y_test))

		if use_ssl:
			print(f"Number of Labeled Points: {NumPointsLst[-1]} | Number of pseudo labels: {num_pseudos} | Number of wrong pseudo labels: {inc_pseudos} | Score: {CurAccuracy[-1]} |")
		else:
			print(f"Number of Labeled Points: {NumPointsLst[-1]} | Score: {CurAccuracy[-1]} |")

	outDict = {
		"num_points": NumPointsLst,
		"scores": CurAccuracy,
		"max_score": max_score
	}

	pickle.dump(outDict, open(f'DLAL_{dataset_name}.pkl', 'wb'))

	return NumPointsLst, CurAccuracy, max_score

def test_model():
	X, y = load_digits(return_X_y=True)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
	model = SVC(probability=True, kernel='linear', max_iter=1000, break_ties=True)
	return active_learn(model, X_train, y_train, X_test, y_test)

def loop_train():
	dlal = DeepLearningActiveLearning(retrain_al = True,
										retrain_ssl = True)
	dlal.trainAL("rank_metric_train.csv")
	dlal.trainSSL("metric_train.csv")

#loop_train()
#test_model()