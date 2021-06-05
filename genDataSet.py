import random
random.seed(786)
import numpy as np
np.random.seed(786)

from sklearn.svm import SVC
from sklearn.datasets import load_digits, load_iris, load_wine, make_classification, make_gaussian_quantiles, make_moons
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import *
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from inf_density import InformationDensity
from egl import expected_gradient
from sklearn.ensemble import RandomForestClassifier

model = SVC(probability=True, kernel='linear')
Cid = InformationDensity()

def percent_change(init_score, new_score):
	return 100 * (new_score - init_score) / init_score

def CountFrequency(my_list): 
	freq = {} 
	for item in my_list: 
		if (item in freq): 
			freq[item] += 1
		else: 
			freq[item] = 1
	return freq

def write_metrics(X, y, OUT, OUT_Rank):

	freq = CountFrequency(y)

	for key, value in freq.items():
		if value > 3:
			continue
		X = X[y != key]
		y = y[y != key]

	y = LabelEncoder().fit_transform(y)

	NumClasses = len(np.unique(y))

	#Cannot train data on only one class...
	if NumClasses == 1:
		return

	NumFeatures = X.shape[1]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

	model.fit(X_train, y_train)
	max_score = model.score(X_test, y_test)

	Cid.fit(X_train)

	#Since we have different init values each time, each run will give different results
	for split in [[0.1], [0.1], [0.1], [0.2], [0.2], [0.2], [False, True], [False, False]]:
	#For now, just use non subset based splits.
	#for split in [False for i in range(10)]:
		if split[0] is False:
			X_rest, X_init, y_rest, y_init = init_split(X_train, y_train, randomSplit=split[0], add_medoids=split[1])
		else:
			X_rest, X_init, y_rest, y_init = init_split(X_train, y_train, randomSplit=True, init_size=split[0])

		def test_dfs(model, X_rest, X_init, y_rest, y_init, bool_lst=None):
			if bool_lst is None:
				bool_lst = np.array([False for i in range(len(X_rest))])
				X_label = X_init
				y_label = y_init
				X_unlabel = X_rest
				y_unlabel = y_init

			counter = 0

			while not all(bool_lst):
				counter += 1

				X_label = np.vstack((X_init, X_rest[bool_lst]))
				y_label = np.append(y_init, y_rest[bool_lst])

				X_unlabel = X_rest[~bool_lst]
				y_unlabel = y_rest[~bool_lst]

				#Fit model on init data
				model.fit(X_label, y_label)
				init_score = model.score(X_test, y_test)

				#If the init score is already high skip it; we wont have much to do here.
				if init_score >= max_score:
					break

				HeuristicMatrix = accumulate_heuristics(model, X_label, y_label, X_unlabel, y_unlabel, Cid, rank_conv = False)
				HeuristicMatrixRank = accumulate_heuristics(model, X_label, y_label, X_unlabel, y_unlabel, Cid, rank_conv = True)
				perfLst = perf_inc_rank(model, X_label, y_label, X_unlabel, y_unlabel, X_test, y_test)
				y_pred = model.predict(X_unlabel)
				CorrectLst = y_pred == y_unlabel

				for index in range(len(X_unlabel)):
					HeuristicLst = HeuristicMatrix[index]
					HeuristicRankLst = HeuristicMatrixRank[index]
					CorrectPrediction = CorrectLst[index]
					if CorrectLst[index]:
						CorrectPrediction = 1
					else:
						CorrectPrediction = 0
					PerfRank = perfLst[index]
					for lst_index in range(len(HeuristicLst)):
						OUT.write(f"{HeuristicLst[lst_index]},")
						OUT_Rank.write(f"{HeuristicRankLst[lst_index]},")
					OUT.write(f"{CorrectPrediction}\n")
					OUT_Rank.write(f"{PerfRank}\n")

				bestIndex = np.argmax(perfLst)

				# "Label" the vector that caused the greatest increase.
				bool_lst[[i for i, n in enumerate(bool_lst) if n == False][bestIndex]] = True

			print(f"Total Number of points labeled: {len(X_init) + counter}")

		#Shuffle the rest of the points in unision
		shufflePerm = np.random.permutation(len(X_rest))
		X_unlabel = X_rest[shufflePerm]
		y_unlabel = y_rest[shufflePerm]
		test_dfs(model, X_unlabel, X_init, y_unlabel, y_init)


#THIS IS FOR VALIDATION / TEST DATASETS
#SHUFFLE AND ONLY USE SMALL SUBSET OF DATA IN THIS CSV

# Open csv write object
with open("metric_valid_test.csv", "w+") as OUT, open("rank_metric_valid_test.csv", "w+") as OUT_Rank:

	#List feature names
	featureLst = [
		"CoLearnConsensus",
		"CoTrainConsensus",
		"Perturbation",
		"LinearRatio",
		"RBFRatio",
		"PolyRatio",
		"SigmoidRatio",
		"ExpectedGradient",
		"qbcConsensus",
		"MinAngle",
		#"Instability",
		"CosDensity",
		"EucDensity",
		"Entropy",
		"MarginConfidence",
		"RatioConfidence",
		"LeastConfidence",
		"NumClasses",
		"NumFeatures",
	]

	for index, feature in enumerate(featureLst):
		OUT.write(f"{feature},")
		OUT_Rank.write(f"{feature},")
	OUT.write("CorrectPrediction\n")
	OUT_Rank.write("PerfRank\n")

	#First, the built-in datasets for sklearn
	X, y = load_digits(return_X_y=True)
	write_metrics(X, y, OUT, OUT_Rank)

	X, y = load_iris(return_X_y=True)
	write_metrics(X, y, OUT, OUT_Rank)
	X, y = load_wine(return_X_y=True)
	write_metrics(X, y, OUT, OUT_Rank)

	#From: https://archive.ics.uci.edu/ml/datasets/Balance+Scale
	X, y = import_data_dataset("balance-scale.data", ["Class", "LW", "LD", "RW", "RD"], catColumns = ['Class'], className='Class')
	write_metrics(X, y, OUT, OUT_Rank)

	#From: https://archive.ics.uci.edu/ml/datasets/car+evaluation
	X, y = import_data_dataset("car.data", ["buying", "maint", "doors", "persons", "lug_boot", "safety", "Class"], catColumns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "Class"], className='Class')
	write_metrics(X, y, OUT, OUT_Rank)

	#From: https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits
	colNames = []
	for i in range(16):
		colNames.append("attribute" + str(i+1))
	colNames.append("Class")
	X, y = import_data_dataset("pendigits.data", colNames, catColumns = [], className="Class")
	write_metrics(X, y, OUT, OUT_Rank)

	#From: https://archive.ics.uci.edu/ml/datasets/mushroom
	colNames = ["Class"]
	for i in range(22):
		colNames.append("attribute" + str(i+1))
	X, y = import_data_dataset("mushroom.data", colNames, catColumns = colNames, className="Class")
	write_metrics(X, y, OUT, OUT_Rank)

	#From: http://archive.ics.uci.edu/ml/datasets/heart+disease
	X, y = import_data_dataset("processed.cleveland.data", ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"], catColumns = [], className="num")
	write_metrics(X, y, OUT, OUT_Rank)

	#From: https://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite)
	colNames = []
	for i in range(36):
		colNames.append("attribute" + str(i+1))
	colNames.append("Class")
	X, y = import_data_dataset("satimage.data", colNames, catColumns = [], className="Class", sep=" ")
	write_metrics(X, y, OUT, OUT_Rank)

	#From: https://archive.ics.uci.edu/ml/datasets/glass+identification
	X, y = import_data_dataset("glass.data", ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"], catColumns = [], dropColumns=["Id"], className="Type")
	write_metrics(X, y, OUT, OUT_Rank)

	#From: https://archive.ics.uci.edu/ml/datasets/image+segmentation
	X, y = import_data_dataset("segmentation.data", ["Class", "REGION-CENTROID-COL","REGION-CENTROID-ROW","REGION-PIXEL-COUNT","SHORT-LINE-DENSITY-5","SHORT-LINE-DENSITY-2","VEDGE-MEAN","VEDGE-SD","HEDGE-MEAN","HEDGE-SD","INTENSITY-MEAN","RAWRED-MEAN","RAWBLUE-MEAN","RAWGREEN-MEAN","EXRED-MEAN","EXBLUE-MEAN","EXGREEN-MEAN","VALUE-MEAN","SATURATION-MEAN","HUE-MEAN"], catColumns = ["Class"], dropColumns=[], className="Class")
	write_metrics(X, y, OUT, OUT_Rank)

	#From: https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Vowel+Recognition+-+Deterding+Data)
	X, y = import_data_dataset("vowel-context.data", ["SplitNum", "Speaker", "Sex", "F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "Class"], catColumns = [], dropColumns=["SplitNum", "Speaker"], className="Class", sep=" ")
	write_metrics(X, y, OUT, OUT_Rank)

	#From: https://archive.ics.uci.edu/ml/datasets/dermatology
	colNames = ["erythema", "scaling", "definite_borders", "itching", "koebner_phenomenon", "polygonal_papules", "follicular_papules", "oral_mucosal_involvement", "knee_elbow_involvement", "scalp_involvement", "family_history", "age", "melanin_incontinence", "eosinophils_infiltrate", "PNL_infiltrate", "fibrosis_papillary_dermis", "exocytosis", "acanthosis", "hyperkeratosis", "parakeratosis", "clubbing_rete", "elongation_rete", "thinning_suprapapillary", "spongiform_pustule", "munro_microabcess", "focal_hypergranulosis", "disappearance_granular", "vacuolisation_basal", "spongiosis", "saw_tooth_retes", "follicular_horn", "perifollicular_parakeratosis", "inflammatory_monoluclear_inflitrate", "band_infiltrate", "Class"]
	X, y = import_data_dataset("dermatology.data", colNames, catColumns = [], dropColumns=[], className="Class")
	write_metrics(X, y, OUT, OUT_Rank)

	#Pcap data from: https://research.aalto.fi/en/datasets/iot-devices-captures
	#Dataset generated using process outlined in patent number: 17139398
	#Internet Of Things (IOT) Device Identification On Corporate Networks Via Adaptive Feature Set To Balance Computational Complexity And Model Bias
	X, y = import_data_dataset("IoT.csv", catColumns = ["Class", "TCPsrc", "TCPdst"], className='Class')
	eval_on_set("IoT", X, y)
	write_metrics(X, y, OUT, OUT_Rank)

"""

#BELOW THIS POINT IS FOR TRAINING DATA.
# Open csv write object

with open("metric_train.csv", "w+") as OUT, open("rank_metric_train.csv", "w+") as OUT_Rank:

	#List feature names
	featureLst = [
		"CoLearnConsensus",
		"CoTrainConsensus",
		"Perturbation",
		"LinearRatio",
		"RBFRatio",
		"PolyRatio",
		"SigmoidRatio",
		"ExpectedGradient",
		"qbcConsensus",
		"MinAngle",
		#"Instability",
		"CosDensity",
		"EucDensity",
		"Entropy",
		"MarginConfidence",
		"RatioConfidence",
		"LeastConfidence",
		"NumClasses",
		"NumFeatures",
	]

	for index, feature in enumerate(featureLst):
		OUT.write(f"{feature},")
		OUT_Rank.write(f"{feature},")
	OUT.write("CorrectPrediction\n")
	OUT_Rank.write("PerfRank\n")

	#Generate numerous example datasets for evaluation
	for num_classes in range(2, 11, 1):
		num_samples = 100*num_classes
		for num_features in range(2, 6):
			max_cluster = int(min((2**num_features)*(1.0/num_classes), 31))
			for num_clusters in range(1, max_cluster):
				for sep in range(2, 13):
					class_sep = sep/10.0
					for hypercube in [True, False]:
						X, y = make_classification(n_samples=num_samples,
													n_features=num_features,
													n_informative=num_features,
													n_redundant=0,
													n_classes=num_classes,
													n_clusters_per_class=num_clusters,
													class_sep=class_sep,
													hypercube=hypercube,
													)
						write_metrics(X, y, OUT, OUT_Rank)
"""
