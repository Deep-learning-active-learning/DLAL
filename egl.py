import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import pandas as pd
import math
from sklearn.preprocessing import LabelBinarizer
import scipy.optimize as opt
from scipy.special import expit
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

#Code from CS230 Logistic Regression with a Neural Network mindset lab
#Modified with inspiration from sklearn's OVR classifier API.
class MultiClassLogReg():
	def __init__(self, lamda=1):
		self.thetas = []
		self.label_binarizer_ = None
		self.lamda = lamda

	def add_bias(self, X):
		return np.hstack((X,np.ones((X.shape[0],1))))

	def sigmoid(self, z):   
		return expit(z)

	def reg_costFunc(self, theta, x, y):
		m = len(x)
		j = (-1/m)*(y.T @ np.log(self.sigmoid(x@ theta)) + (1 - y.T) @ np.log(1 - self.sigmoid(x@theta)))
		reg = (self.lamda/(2*m))*(theta[1:].T @ theta[1:])
		j = j + reg
		return j

	def gradient_estimator(self, theta, x, y):
		m = len(x)
		grad = np.zeros([m,1])
		grad = (1/m) * x.T @ (self.sigmoid(x @ theta) - y)
		grad[1:] = grad[1:] + (self.lamda / m) * theta[1:]
		return grad

	def magnitude_gradient(self, X):
		X_norm = np.linalg.norm(X, axis=1)
		gradMag = np.array([np.abs(self.sigmoid(X @ self.thetas[index]) - self.classes_[index])*X_norm for index in range(len(self.classes_))]).T
		return gradMag

	def train_estimator(self, X, column):
		theta = np.random.randn(X.shape[1],1)*0.01
		output = opt.fmin_tnc(func = self.reg_costFunc, x0 = theta.flatten(), fprime = self.gradient_estimator, args = (X, column.flatten()), messages=0)
		theta = output[0]
		return theta

	def predict_estimator(self, X, theta):
		return self.sigmoid(X @ theta)

	def fit(self, X, y):
		X = self.add_bias(X)
		self.label_binarizer_ = LabelBinarizer(sparse_output=True)
		Y = self.label_binarizer_.fit_transform(y)
		Y = Y.tocsc()
		self.classes_ = self.label_binarizer_.classes_
		columns = (col.toarray().ravel() for col in Y.T)
		for i, column in enumerate(columns):
			#train on X, column
			self.thetas.append(self.train_estimator(X, column))

	def predict_proba(self, X):
		Y = np.array([self.predict_estimator(X, theta) for theta in self.thetas]).T
		Y /= np.sum(Y, axis=1)[:, np.newaxis]
		return Y

	def expected_gradient_length(self, X):
		X = self.add_bias(X)
		grads = np.sum(np.multiply(self.predict_proba(X), self.magnitude_gradient(X)), axis=1)
		return grads/max(grads)

	def predict(self, X):
		X = self.add_bias(X)
		return np.argmax(self.predict_proba(X), axis=1)

def expected_gradient(X_label, y_label, X_unlabel):
	model = MultiClassLogReg()
	model.fit(X_label, y_label)
	return model.expected_gradient_length(X_unlabel)

def main():
	X, Y = load_digits(return_X_y=True)
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, stratify=Y)
	model = MultiClassLogReg()
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
