from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_FAIL
from hyperopt.pyll import scope as ho_scope
from hyperopt.pyll.stochastic import sample as ho_sample
import pickle
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import warnings
warnings.filterwarnings("ignore")
import traceback
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Nadam, Adam
import gc
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.ops import math_ops

# Weighted version of log_loss; allows us to weigh precision over recall.
# Used to train SSL model.
def log_loss(y_true, y_pred, weight=1):
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

def residual_block(hidden, layer_dim=18, kernel_initializer='he_normal', activation='selu', l1_lambda=1e-5, l2_lambda=1e-4, batchNorm = True):
	#hidden = Dropout(0.1)(hidden)
	hidden_new = Dense(layer_dim, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(hidden)
	try:
		out = Add()([hidden, hidden_new])
	except:
		#Dimension mismatch; User intended to skip Residual and simply use Plain.
		out = hidden_new
	out = Activation(activation)(out)
	if batchNorm:
		out = BatchNormalization()(out)
	return out

def compile_model(_model = "VPLNN", first_act='selu', first_size=36, first_num=10, sec_act='selu', sec_size=36, sec_num=10, third_act='selu', third_size=36, third_num=10, optimizer='adam', learning_rate=0.001, l1_lambda=1e-5, l2_lambda=1e-4, batchNorm = True, tau=1):
	#Converts activation type into ideal initializer
	actToInit = {
		"selu": 'he_normal',
		"relu": 'he_normal',
		"tanh": 'glorot_uniform',
		"sigmoid": 'glorot_uniform'
	}

	# Create model
	inputLayer = Input(shape=(18,))
	hidden = residual_block(inputLayer, layer_dim=first_size, kernel_initializer=actToInit[first_act], activation=first_act, l1_lambda=l1_lambda, l2_lambda=l2_lambda, batchNorm = batchNorm)
	for i in range(first_num - 1):
		hidden = residual_block(hidden, layer_dim=first_size, kernel_initializer=actToInit[first_act], activation=first_act, l1_lambda=l1_lambda, l2_lambda=l2_lambda, batchNorm = batchNorm)
	for i in range(sec_num):
		hidden = residual_block(hidden, layer_dim=sec_size, kernel_initializer=actToInit[sec_act], activation=sec_act, l1_lambda=l1_lambda, l2_lambda=l2_lambda, batchNorm = batchNorm)
	for i in range(third_num):
		hidden = residual_block(hidden, layer_dim=third_size, kernel_initializer=actToInit[third_act], activation=third_act, l1_lambda=l1_lambda, l2_lambda=l2_lambda, batchNorm = batchNorm)

	if _model == "VPLNN":
		out = Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid')(hidden)
	elif _model == "VQNN":
		out = Dense(1, kernel_initializer='he_normal', activation='relu')(hidden)

	model = Model(inputs=inputLayer, outputs=out)

	if optimizer == 'adam':
		opt = Adam(learning_rate)
	elif optimizer == 'nadam':
		opt = Nadam(learning_rate)
	else:
		opt = optimizer

	if _model == "VPLNN":
		model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])
	elif _model == "VQNN":
		mod_mse = lambda y_true, y_pred: mse(y_true, y_pred, tau=tau)
		model.compile(loss=mod_mse, optimizer=opt)

	return model

def create_model(_model = "VPLNN"):
	space = {
		'clf': {
			'first_act': hp.choice('first_act', ['selu', 'relu', 'tanh', 'sigmoid']),
			'first_size': hp.choice('first_size', range(1, 37)),
			'first_num': hp.choice('first_num', range(1, 11)),

			'sec_act': hp.choice('sec_act', ['selu', 'relu', 'tanh', 'sigmoid']),
			'sec_size': hp.choice('sec_size', range(1, 37)),
			'sec_num': hp.choice('sec_num', range(0, 11)),

			'third_act': hp.choice('third_act', ['selu', 'relu', 'tanh', 'sigmoid']),
			'third_size': hp.choice('third_size', range(1, 37)),
			'third_num': hp.choice('third_num', range(0, 11)),

			'optimizer': hp.choice('optimize', ['adam', 'nadam']),
			'learning_rate': hp.loguniform("learning_rate", np.log(0.00001), np.log(0.5)),

			'l1_lambda': hp.loguniform("l1_lambda", np.log(0.00000001), np.log(0.5)),
			'l2_lambda': hp.loguniform("l2_lambda", np.log(0.00000001), np.log(0.5)),

			'batchNorm': hp.choice('batchNorm', [True, False]),
		},
		'epochs': 1000,
		'batch_size': 4096,
		'verbose': 1,
		'callbacks': [EarlyStopping(monitor='loss', patience=100, verbose=0, restore_best_weights=True)],
		'shuffle': True
	}

	if _model == "VQNN":
		space['clf']['tau'] = hp.choice('tau', range(1, 1001))
	else:
		space['batch_size'] = 16384

	clfDict = {}

	with open(f"{_model}.hyperopt", "rb") as f:
		trials = pickle.load(f)

		best_trial = {}
		for key in trials.best_trial['misc']['vals']:
			val = trials.best_trial['misc']['vals'][key]
			best_trial[key] = val[0]

		args = space_eval(space, best_trial)

		clfDict = args['clf']

	#Add current model info
	clfDict["_model"] = _model

	print(clfDict)

	# Model setup
	model = compile_model(**clfDict)

	gc.collect()

	return model

create_model(_model = "VPLNN")