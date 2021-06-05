import numpy as np
from numpy import matlib as mb
import sklearn
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

"""
Derived from:
https://github.com/YazhouTUD/MVAL/blob/MVAL/MVAL_logistic_multi.m
"""
def MVAL(model, X_label, X_unlabel, y_label, y_unlabel):

	#Train the model on labeled data
	model.fit(X_label, y_label)

	#Get posterior probabilities
	probs = model.predict_proba(X_unlabel)

	classes = np.unique(y_label)
	K = len(classes)

	sto = []

	#Iterate through all unlabeled values
	for index, X_val in tqdm(enumerate(X_unlabel)):
		X_cur = X_val.reshape(1, -1)
		y_cur = y_unlabel[index]

		#Add to "labeled" data
		X_new = np.vstack((X_label, X_cur))

		prb = None

		for curClass in classes:
			y_new = np.append(y_label, curClass)
			model.fit(X_new, y_new)
			probs = model.predict_proba(X_unlabel)

			probs_new = 1.0/(1+np.exp(-1*probs))

			if prb is None:
				prb = probs_new
			else:
				prb = np.append(prb, probs_new, axis=1)

		prb = np.array(prb)

		sto.append(prb)

	sto = np.array(sto)

	#Compute sizes M and N
	N = sto.shape[0]
	M = sto.shape[1]

	Amx = np.zeros((N,M,K,K))

	for i in range(K):
		Amx[:,:,i,:] = sto[:,:,(i)*K:(i+1)*K]

	ABmx = np.zeros((N*K,N,K))
	for i in range(K):
		ABmx[i*N:(i+1)*N,:,:] = np.squeeze(Amx[:,:,i,:])

	var11 = np.squeeze(np.var(ABmx, axis=0))

	if N == 1:
		var1 = np.sum(var11)
	else:
		var1 = np.sum(var11, axis=1)

	CDmx = np.zeros((N,N*K,K))
	for i in range(K-1):
		CDmx[:,i*N:(i+1)*N,:] = np.squeeze(Amx[:,:,i+1,:]) - np.squeeze(Amx[:,:,i,:])
	CDmx[:,((K-1)*N):(K)*N,:] = np.squeeze(Amx[:,:,0,:]) - np.squeeze(Amx[:,:,K-1,:])

	var22 = np.squeeze(np.var(CDmx, axis=1))

	if N == 1:
		var2 = np.sum(var22)
	else:
		var2 = np.sum(var22, axis=1)

	lsort = np.sort(probs, axis=1)
	bsb = lsort[:,-1] - lsort[:,-2]
	ent = np.exp(-bsb)

	#Scale matrix
	wCDmx = CDmx
	for j in range(len(ent)):
		wCDmx[:,j,:] = ent[j]*CDmx[:,j,:]

	var44 = np.squeeze(np.var(CDmx, axis=1))

	if N == 1:
		V2 = np.sum(var44)
	else:
		V2 = np.sum(var22, axis=1)

	var33 = var11
	for j in range(len(ent)):
		if N == 1:
			var33[j] = ent[j]*var11[j]
		else:
			var33[j,:] = ent[j]*var11[j,:]

	if N == 1:
		V1 = np.sum(var33)
	else:
		V1 = np.sum(var33, axis=1)

	values = np.multiply(V1,V2)

	try:
		return values/max(values)
	except:
		return [1.0]

def main():
	X, y = load_digits(return_X_y=True)
	model = SVC(probability=True)
	#model = RandomForestClassifier()

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99)

	obj = MVAL(model, X_train, X_test, y_train, y_test)
	print(obj)
