#import random
#random.seed(786)
#import numpy as np
#np.random.seed(786)

from utils import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from mval import MVAL
from sklearn.decomposition import PCA
import pickle
from inf_density import InformationDensity
from DLAL import active_learn as DLAL

#We will use the multi-class LogReg which acts akin to softmax
from sklearn.linear_model import LogisticRegression as SoftMax

def eval_probs(model, X_unlabel, is_lin_svm=True):
	try:
		probs = model.predict_proba(X_unlabel)
	except:
		probs = model.decision_function(X_unlabel)

	if len(probs.shape) != 2:
		probs2 = np.zeros((len(probs), 2))
		if is_lin_svm:
			probs2[:,0] = -probs
		else:
			probs2[:,0] = 1 - probs
		probs2[:,1] = probs
		probs = probs2

	return probs

def Variance_Maximization_Al(model, X_train, y_train, X_test, y_test, dataset_name, num_init=11):
	#Like in the original paper, we apply PCA + undersample the training set due to algorithm's computational complexity.
	#Unfortunately, this is required for our test bench to not kill processes due to memory limits.
	try:
		pca = PCA(n_components=10)
		pca.fit(X_train)
		X_train = pca.transform(X_train)
		X_test = pca.transform(X_test)
	except:
		pass

	numTrain = len(X_train)

	if numTrain > 1000:
		cur_perm = np.random.permutation(numTrain)
		X_train = X_train[cur_perm][:1000]
		y_train = y_train[cur_perm][:1000]

	#This represents the maximum score we can hope to achieve with the given model.
	model.fit(X_train, y_train)
	max_score = model.score(X_test, y_test)

	print(f"Best Score: {max_score} | Num Training Points: {len(X_train)}")

	#To match the number of initial values selected via DLAL, take first num_init into labeled set
	valid_split = False
	while not valid_split:
		try:
			cur_perm = np.random.permutation(len(X_train))
			X_train = X_train[cur_perm]
			y_train = y_train[cur_perm]

			X_label = X_train[:num_init]
			y_label = y_train[:num_init]
			X_unlabel = X_train[num_init:]
			y_unlabel = y_train[num_init:]
			model.fit(X_label, y_label)

			valid_split = True
		except:
			pass

	NumPointsLst = [len(X_label)]
	CurAccuracy = [model.score(X_test, y_test)]
	print(f"Number of Labeled Points: {NumPointsLst[-1]} Score: {CurAccuracy[-1]}")

	while CurAccuracy[-1] < max_score:
		var_lst = MVAL(model, X_label, X_unlabel, y_label, y_unlabel)

		indices = [np.argmax(var_lst)]
		X_unlabel, y_unlabel, X_label, y_label = migrate_vectors(X_unlabel, y_unlabel, X_label, y_label, indices)
		model.fit(X_label, y_label)
		NumPointsLst.append(len(X_label))
		CurAccuracy.append(model.score(X_test, y_test))
		print(f"Number of Labeled Points: {NumPointsLst[-1]} | Score: {CurAccuracy[-1]} |")

	outDict = {
		"num_points": NumPointsLst,
		"scores": CurAccuracy,
		"max_score": max_score
	}

	pickle.dump(outDict, open(f"MVAL_{dataset_name}.pkl", 'wb'))

	return NumPointsLst, CurAccuracy, max_score

def Entropy_Al(model, X_train, y_train, X_test, y_test, dataset_name, num_init=11):

	#This represents the maximum score we can hope to achieve with the given model.
	model.fit(X_train, y_train)
	max_score = model.score(X_test, y_test)

	print(f"Best Score: {max_score} | Num Training Points: {len(X_train)}")

	#To match the number of initial values selected via DLAL, take first num_init into labeled set
	valid_split = False
	while not valid_split:
		try:
			cur_perm = np.random.permutation(len(X_train))
			X_train = X_train[cur_perm]
			y_train = y_train[cur_perm]

			X_label = X_train[:num_init]
			y_label = y_train[:num_init]
			X_unlabel = X_train[num_init:]
			y_unlabel = y_train[num_init:]
			model.fit(X_label, y_label)
			valid_split = True
		except:
			pass

	NumPointsLst = [len(X_label)]
	CurAccuracy = [model.score(X_test, y_test)]
	print(f"Number of Labeled Points: {NumPointsLst[-1]} Score: {CurAccuracy[-1]}")

	while CurAccuracy[-1] < max_score:
		EntropyLst = []
		for index, X_val in enumerate(X_unlabel):
			X_cur = X_val.reshape(1, -1)

			probLst = eval_probs(model, X_unlabel)[0]
			
			#Get uncertainty metrics
			EntropyLst.append(entropy_score(probLst))

		indices = [np.argmax(EntropyLst)]
		X_unlabel, y_unlabel, X_label, y_label = migrate_vectors(X_unlabel, y_unlabel, X_label, y_label, indices)
		model.fit(X_label, y_label)
		NumPointsLst.append(len(X_label))
		CurAccuracy.append(model.score(X_test, y_test))
		print(f"Number of Labeled Points: {NumPointsLst[-1]} | Score: {CurAccuracy[-1]} |")

	outDict = {
		"num_points": NumPointsLst,
		"scores": CurAccuracy,
		"max_score": max_score
	}

	pickle.dump(outDict, open(f"EAL_{dataset_name}.pkl", 'wb'))

	return NumPointsLst, CurAccuracy, max_score

def Entropy_InfDensity_Al(model, X_train, y_train, X_test, y_test, dataset_name, num_init=11):

	#This represents the maximum score we can hope to achieve with the given model.
	model.fit(X_train, y_train)
	max_score = model.score(X_test, y_test)

	Cid = InformationDensity()
	Cid.fit(X_train)

	print(f"Best Score: {max_score} | Num Training Points: {len(X_train)}")

	#To match the number of initial values selected via DLAL, take first num_init into labeled set
	valid_split = False
	while not valid_split:
		try:
			cur_perm = np.random.permutation(len(X_train))
			X_train = X_train[cur_perm]
			y_train = y_train[cur_perm]

			X_label = X_train[:num_init]
			y_label = y_train[:num_init]
			X_unlabel = X_train[num_init:]
			y_unlabel = y_train[num_init:]
			model.fit(X_label, y_label)
			valid_split = True
		except:
			pass

	NumPointsLst = [len(X_label)]
	CurAccuracy = [model.score(X_test, y_test)]
	print(f"Number of Labeled Points: {NumPointsLst[-1]} Score: {CurAccuracy[-1]}")

	while CurAccuracy[-1] < max_score:
		EntropyLst = []
		for index, X_val in enumerate(X_unlabel):
			X_cur = X_val.reshape(1, -1)

			probLst = eval_probs(model, X_unlabel)[0]
			
			#Get uncertainty metrics
			EntropyLst.append(entropy_score(probLst)*Cid.density(X_cur)[1])

		indices = [np.argmax(EntropyLst)]
		X_unlabel, y_unlabel, X_label, y_label = migrate_vectors(X_unlabel, y_unlabel, X_label, y_label, indices)
		model.fit(X_label, y_label)
		NumPointsLst.append(len(X_label))
		CurAccuracy.append(model.score(X_test, y_test))
		print(f"Number of Labeled Points: {NumPointsLst[-1]} | Score: {CurAccuracy[-1]} |")

	outDict = {
		"num_points": NumPointsLst,
		"scores": CurAccuracy,
		"max_score": max_score
	}

	pickle.dump(outDict, open(f"EIDAL_{dataset_name}.pkl", 'wb'))

	return NumPointsLst, CurAccuracy, max_score

def MarginConfidence_Al(model, X_train, y_train, X_test, y_test, dataset_name, num_init=11):

	#This represents the maximum score we can hope to achieve with the given model.
	model.fit(X_train, y_train)
	max_score = model.score(X_test, y_test)

	print(f"Best Score: {max_score} | Num Training Points: {len(X_train)}")

	#To match the number of initial values selected via DLAL, take first num_init into labeled set
	valid_split = False
	while not valid_split:
		try:
			cur_perm = np.random.permutation(len(X_train))
			X_train = X_train[cur_perm]
			y_train = y_train[cur_perm]

			X_label = X_train[:num_init]
			y_label = y_train[:num_init]
			X_unlabel = X_train[num_init:]
			y_unlabel = y_train[num_init:]
			model.fit(X_label, y_label)
			valid_split = True
		except:
			pass

	NumPointsLst = [len(X_label)]
	CurAccuracy = [model.score(X_test, y_test)]
	print(f"Number of Labeled Points: {NumPointsLst[-1]} Score: {CurAccuracy[-1]}")

	while CurAccuracy[-1] < max_score:

		probs = np.sort(model.predict_proba(X_unlabel), axis=1)
		diff = (probs[:,-1] - probs[:,-2])
		MarginConfidenceLst = 1 - diff

		indices = [np.argmax(MarginConfidenceLst)]
		X_unlabel, y_unlabel, X_label, y_label = migrate_vectors(X_unlabel, y_unlabel, X_label, y_label, indices)
		model.fit(X_label, y_label)
		NumPointsLst.append(len(X_label))
		CurAccuracy.append(model.score(X_test, y_test))
		print(f"Number of Labeled Points: {NumPointsLst[-1]} | Score: {CurAccuracy[-1]} |")

	outDict = {
		"num_points": NumPointsLst,
		"scores": CurAccuracy,
		"max_score": max_score
	}

	pickle.dump(outDict, open(f"MCAL_{dataset_name}.pkl", 'wb'))

	return NumPointsLst, CurAccuracy, max_score

def RatioConfidence_Al(model, X_train, y_train, X_test, y_test, dataset_name, num_init=11):

	#This represents the maximum score we can hope to achieve with the given model.
	model.fit(X_train, y_train)
	max_score = model.score(X_test, y_test)

	print(f"Best Score: {max_score} | Num Training Points: {len(X_train)}")

	#To match the number of initial values selected via DLAL, take first num_init into labeled set
	valid_split = False
	while not valid_split:
		try:
			cur_perm = np.random.permutation(len(X_train))
			X_train = X_train[cur_perm]
			y_train = y_train[cur_perm]

			X_label = X_train[:num_init]
			y_label = y_train[:num_init]
			X_unlabel = X_train[num_init:]
			y_unlabel = y_train[num_init:]
			model.fit(X_label, y_label)
			valid_split = True
		except:
			pass

	NumPointsLst = [len(X_label)]
	CurAccuracy = [model.score(X_test, y_test)]
	print(f"Number of Labeled Points: {NumPointsLst[-1]} Score: {CurAccuracy[-1]}")

	while CurAccuracy[-1] < max_score:
		probs = np.sort(model.predict_proba(X_unlabel), axis=1)
		RatioConfidenceLst = np.divide(probs[:,-2], probs[:,-1])

		indices = [np.argmax(RatioConfidenceLst)]
		X_unlabel, y_unlabel, X_label, y_label = migrate_vectors(X_unlabel, y_unlabel, X_label, y_label, indices)
		model.fit(X_label, y_label)
		NumPointsLst.append(len(X_label))
		CurAccuracy.append(model.score(X_test, y_test))
		print(f"Number of Labeled Points: {NumPointsLst[-1]} | Score: {CurAccuracy[-1]} |")

	outDict = {
		"num_points": NumPointsLst,
		"scores": CurAccuracy,
		"max_score": max_score
	}

	pickle.dump(outDict, open(f"RCAL_{dataset_name}.pkl", 'wb'))

	return NumPointsLst, CurAccuracy, max_score

def LeastConfidence_Al(model, X_train, y_train, X_test, y_test, dataset_name, num_init=11):

	#This represents the maximum score we can hope to achieve with the given model.
	model.fit(X_train, y_train)
	max_score = model.score(X_test, y_test)

	print(f"Best Score: {max_score} | Num Training Points: {len(X_train)}")

	#To match the number of initial values selected via DLAL, take first num_init into labeled set
	valid_split = False
	while not valid_split:
		try:
			cur_perm = np.random.permutation(len(X_train))
			X_train = X_train[cur_perm]
			y_train = y_train[cur_perm]

			X_label = X_train[:num_init]
			y_label = y_train[:num_init]
			X_unlabel = X_train[num_init:]
			y_unlabel = y_train[num_init:]
			model.fit(X_label, y_label)
			valid_split = True
		except:
			pass

	NumPointsLst = [len(X_label)]
	CurAccuracy = [model.score(X_test, y_test)]
	print(f"Number of Labeled Points: {NumPointsLst[-1]} Score: {CurAccuracy[-1]}")

	while CurAccuracy[-1] < max_score:
		LeastConfidenceLst = np.amin(model.predict_proba(X_unlabel), axis=1)

		indices = [np.argmax(LeastConfidenceLst)]

		indices = [np.argmax(LeastConfidenceLst)]
		X_unlabel, y_unlabel, X_label, y_label = migrate_vectors(X_unlabel, y_unlabel, X_label, y_label, indices)
		model.fit(X_label, y_label)
		NumPointsLst.append(len(X_label))
		CurAccuracy.append(model.score(X_test, y_test))
		print(f"Number of Labeled Points: {NumPointsLst[-1]} | Score: {CurAccuracy[-1]} |")

	outDict = {
		"num_points": NumPointsLst,
		"scores": CurAccuracy,
		"max_score": max_score
	}

	pickle.dump(outDict, open(f"LCAL_{dataset_name}.pkl", 'wb'))

	return NumPointsLst, CurAccuracy, max_score

def RandomSelection(model, X_train, y_train, X_test, y_test, dataset_name, num_init=11):

	#This represents the maximum score we can hope to achieve with the given model.
	model.fit(X_train, y_train)
	max_score = model.score(X_test, y_test)

	print(f"Best Score: {max_score} | Num Training Points: {len(X_train)}")

	#To match the number of initial values selected via DLAL, take first num_init into labeled set
	valid_split = False
	while not valid_split:
		try:
			cur_perm = np.random.permutation(len(X_train))
			X_train = X_train[cur_perm]
			y_train = y_train[cur_perm]

			X_label = X_train[:num_init]
			y_label = y_train[:num_init]
			X_unlabel = X_train[num_init:]
			y_unlabel = y_train[num_init:]
			model.fit(X_label, y_label)
			valid_split = True
		except:
			pass

	NumPointsLst = [len(X_label)]
	CurAccuracy = [model.score(X_test, y_test)]
	print(f"Number of Labeled Points: {NumPointsLst[-1]} Score: {CurAccuracy[-1]}")

	while CurAccuracy[-1] < max_score:
		indices = [random.choice(list(enumerate(X_unlabel)))[0]]
		X_unlabel, y_unlabel, X_label, y_label = migrate_vectors(X_unlabel, y_unlabel, X_label, y_label, indices)
		model.fit(X_label, y_label)
		NumPointsLst.append(len(X_label))
		CurAccuracy.append(model.score(X_test, y_test))
		print(f"Number of Labeled Points: {NumPointsLst[-1]} | Score: {CurAccuracy[-1]} |")

	outDict = {
		"num_points": NumPointsLst,
		"scores": CurAccuracy,
		"max_score": max_score
	}

	pickle.dump(outDict, open(f"RS_{dataset_name}.pkl", 'wb'))

	return NumPointsLst, CurAccuracy, max_score

def LinearBound_Al(model, X_train, y_train, X_test, y_test, dataset_name, num_init=11):

	#This represents the maximum score we can hope to achieve with the given model.
	model.fit(X_train, y_train)
	max_score = model.score(X_test, y_test)

	print(f"Best Score: {max_score} | Num Training Points: {len(X_train)}")

	#To match the number of initial values selected via DLAL, take first num_init into labeled set
	valid_split = False
	while not valid_split:
		try:
			cur_perm = np.random.permutation(len(X_train))
			X_train = X_train[cur_perm]
			y_train = y_train[cur_perm]

			X_label = X_train[:num_init]
			y_label = y_train[:num_init]
			X_unlabel = X_train[num_init:]
			y_unlabel = y_train[num_init:]
			model.fit(X_label, y_label)
			valid_split = True
		except:
			pass

	NumPointsLst = [len(X_label)]
	CurAccuracy = [model.score(X_test, y_test)]
	print(f"Number of Labeled Points: {NumPointsLst[-1]} Score: {CurAccuracy[-1]}")

	while CurAccuracy[-1] < max_score:
		LinearBoundLst = get_bound_list(X_label, X_unlabel, y_label, y_unlabel, kernel="linear")
		indices = [np.argmin(LinearBoundLst)]
		X_unlabel, y_unlabel, X_label, y_label = migrate_vectors(X_unlabel, y_unlabel, X_label, y_label, indices)
		model.fit(X_label, y_label)
		NumPointsLst.append(len(X_label))
		CurAccuracy.append(model.score(X_test, y_test))
		print(f"Number of Labeled Points: {NumPointsLst[-1]} | Score: {CurAccuracy[-1]} |")

	outDict = {
		"num_points": NumPointsLst,
		"scores": CurAccuracy,
		"max_score": max_score
	}

	pickle.dump(outDict, open(f"LBAL_{dataset_name}.pkl", 'wb'))

	return NumPointsLst, CurAccuracy, max_score

def EGL_Al(X_train, y_train, X_test, y_test, dataset_name, num_init=11):

	model = SoftMax()

	#This represents the maximum score we can hope to achieve with the given model.
	model.fit(X_train, y_train)
	max_score = model.score(X_test, y_test)

	print(f"Best Score: {max_score} | Num Training Points: {len(X_train)}")

	#To match the number of initial values selected via DLAL, take first num_init into labeled set
	valid_split = False
	while not valid_split:
		try:
			cur_perm = np.random.permutation(len(X_train))
			X_train = X_train[cur_perm]
			y_train = y_train[cur_perm]

			X_label = X_train[:num_init]
			y_label = y_train[:num_init]
			X_unlabel = X_train[num_init:]
			y_unlabel = y_train[num_init:]
			model.fit(X_label, y_label)
			valid_split = True
		except:
			pass

	NumPointsLst = [len(X_label)]
	CurAccuracy = [model.score(X_test, y_test)]
	print(f"Number of Labeled Points: {NumPointsLst[-1]} Score: {CurAccuracy[-1]}")

	while CurAccuracy[-1] < max_score:
		gradLst = expected_gradient(X_label, y_label, X_unlabel)
		indices = [np.argmax(gradLst)]
		X_unlabel, y_unlabel, X_label, y_label = migrate_vectors(X_unlabel, y_unlabel, X_label, y_label, indices)
		model.fit(X_label, y_label)
		NumPointsLst.append(len(X_label))
		CurAccuracy.append(model.score(X_test, y_test))
		print(f"Number of Labeled Points: {NumPointsLst[-1]} | Score: {CurAccuracy[-1]} |")

	outDict = {
		"num_points": NumPointsLst,
		"scores": CurAccuracy,
		"max_score": max_score
	}

	pickle.dump(outDict, open(f"EGLAL_{dataset_name}.pkl", 'wb'))

	return NumPointsLst, CurAccuracy, max_score

def eval_on_set(data_name, X, y, num_init=None):
	print(f"Evaluating: {data_name}")
	model = SVC(kernel='linear', probability=True)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
	if num_init is None:
		X_unlabel, X_label, y_unlabel, y_label = init_split(X_train, y_train, randomSplit=False, add_medoids=True)
		num_init = len(X_label)

	DLAL(model, X_train, y_train, X_test, y_test, dataset_name=data_name, use_ssl=False)

	EGL_Al(X_train, y_train, X_test, y_test, dataset_name=data_name, num_init=num_init)

	LinearBound_Al(model, X_train, y_train, X_test, y_test, dataset_name=data_name, num_init=num_init)

	Entropy_InfDensity_Al(model, X_train, y_train, X_test, y_test, dataset_name=data_name, num_init=num_init)

	Entropy_Al(model, X_train, y_train, X_test, y_test, dataset_name=data_name, num_init=num_init)

	RandomSelection(model, X_train, y_train, X_test, y_test, dataset_name=data_name, num_init=num_init)

	MarginConfidence_Al(model, X_train, y_train, X_test, y_test, dataset_name=data_name, num_init=num_init)
	RatioConfidence_Al(model, X_train, y_train, X_test, y_test, dataset_name=data_name, num_init=num_init)
	LeastConfidence_Al(model, X_train, y_train, X_test, y_test, dataset_name=data_name, num_init=num_init)

	Variance_Maximization_Al(model, X_train, y_train, X_test, y_test, dataset_name=data_name, num_init=num_init)

def compute_baseline():

	X, y = load_digits(return_X_y=True)
	eval_on_set("Digits", X, y)

	X, y = load_iris(return_X_y=True)
	eval_on_set("Iris", X, y)

	X, y = load_wine(return_X_y=True)
	eval_on_set("Wine", X, y)

	#From: https://archive.ics.uci.edu/ml/datasets/Balance+Scale
	X, y = import_data_dataset("balance-scale.data", ["Class", "LW", "LD", "RW", "RD"], catColumns = ['Class'], className='Class')
	eval_on_set("Balance", X, y)

	#From: https://archive.ics.uci.edu/ml/datasets/car+evaluation
	X, y = import_data_dataset("car.data", ["buying", "maint", "doors", "persons", "lug_boot", "safety", "Class"], catColumns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "Class"], className='Class')
	eval_on_set("Car", X, y)

	#From: https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits
	colNames = []
	for i in range(16):
		colNames.append("attribute" + str(i+1))
	colNames.append("Class")
	X, y = import_data_dataset("pendigits.data", colNames, catColumns = [], className="Class")
	eval_on_set("Pen", X, y)

	#From: https://archive.ics.uci.edu/ml/datasets/mushroom
	colNames = ["Class"]
	for i in range(22):
		colNames.append("attribute" + str(i+1))
	X, y = import_data_dataset("mushroom.data", colNames, catColumns = colNames, className="Class")
	eval_on_set("Mushroom", X, y)

	#From: http://archive.ics.uci.edu/ml/datasets/heart+disease
	X, y = import_data_dataset("processed.cleveland.data", ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"], catColumns = [], className="num")
	eval_on_set("Heart", X, y)

	#From: https://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite)
	colNames = []
	for i in range(36):
		colNames.append("attribute" + str(i+1))
	colNames.append("Class")
	X, y = import_data_dataset("satimage.data", colNames, catColumns = [], className="Class", sep=" ")
	eval_on_set("SatImage", X, y)

	#From: https://archive.ics.uci.edu/ml/datasets/glass+identification
	X, y = import_data_dataset("glass.data", ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"], catColumns = [], dropColumns=["Id"], className="Type")
	eval_on_set("Glass", X, y)

	#From: https://archive.ics.uci.edu/ml/datasets/image+segmentation
	X, y = import_data_dataset("segmentation.data", ["Class", "REGION-CENTROID-COL","REGION-CENTROID-ROW","REGION-PIXEL-COUNT","SHORT-LINE-DENSITY-5","SHORT-LINE-DENSITY-2","VEDGE-MEAN","VEDGE-SD","HEDGE-MEAN","HEDGE-SD","INTENSITY-MEAN","RAWRED-MEAN","RAWBLUE-MEAN","RAWGREEN-MEAN","EXRED-MEAN","EXBLUE-MEAN","EXGREEN-MEAN","VALUE-MEAN","SATURATION-MEAN","HUE-MEAN"], catColumns = ["Class"], dropColumns=[], className="Class")
	eval_on_set("Segmentation", X, y)

	#From: https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Vowel+Recognition+-+Deterding+Data)
	X, y = import_data_dataset("vowel-context.data", ["SplitNum", "Speaker", "Sex", "F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "Class"], catColumns = [], dropColumns=["SplitNum", "Speaker"], className="Class", sep=" ")
	eval_on_set("Vowel", X, y)

	#From: https://archive.ics.uci.edu/ml/datasets/dermatology
	colNames = ["erythema", "scaling", "definite_borders", "itching", "koebner_phenomenon", "polygonal_papules", "follicular_papules", "oral_mucosal_involvement", "knee_elbow_involvement", "scalp_involvement", "family_history", "age", "melanin_incontinence", "eosinophils_infiltrate", "PNL_infiltrate", "fibrosis_papillary_dermis", "exocytosis", "acanthosis", "hyperkeratosis", "parakeratosis", "clubbing_rete", "elongation_rete", "thinning_suprapapillary", "spongiform_pustule", "munro_microabcess", "focal_hypergranulosis", "disappearance_granular", "vacuolisation_basal", "spongiosis", "saw_tooth_retes", "follicular_horn", "perifollicular_parakeratosis", "inflammatory_monoluclear_inflitrate", "band_infiltrate", "Class"]
	X, y = import_data_dataset("dermatology.data", colNames, catColumns = [], dropColumns=[], className="Class")
	eval_on_set("Dermatology", X, y)

	#Pcap data from: https://research.aalto.fi/en/datasets/iot-devices-captures
	#Dataset generated using process outlined in patent number: 17139398
	#Internet Of Things (IOT) Device Identification On Corporate Networks Via Adaptive Feature Set To Balance Computational Complexity And Model Bias
	X, y = import_data_dataset("IoT.csv", catColumns = ["Class", "TCPsrc", "TCPdst"], className='Class')
	eval_on_set("IoT", X, y)

compute_baseline()
