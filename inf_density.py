from modAL.density import information_density
import numpy as np
from numpy import matlib as mb
import sklearn
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from utils import *

class InformationDensity():
	def __init__(self):
		self.densitySinCache = {}
		self.densityEucCache = {}

	def fit(self, X):
		cosine_density = information_density(X, 'cosine')
		euclidean_density = information_density(X, 'euclidean')
		for index, X_val in enumerate(X):
			X_cur = X_val.reshape(1, -1)
			key = str(X_cur)
			self.densitySinCache[key] = cosine_density[index]
			self.densityEucCache[key] = euclidean_density[index]

	#X_cur is assumed to already have been reshaped
	def density(self, X_cur):
		key = str(X_cur)
		if key not in self.densitySinCache:
			return 0.0, 0.0
		return self.densitySinCache[key], self.densityEucCache[key]

def main():
	X, y = load_digits(return_X_y=True)
	Cid = InformationDensity()
	Cid.fit(X)
	for index1, X_val1 in enumerate(X):
		X_cur1 = X_val1.reshape(1, -1)
		print(f"IDS: {Cid.density(X_cur1)}")
