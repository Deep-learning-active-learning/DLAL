import scipy.stats
import numpy as np
#pip install scikit-learn-extra
from sklearn_extra.cluster import KMedoids
from sklearn.linear_model import LogisticRegression
from sklearn import mixture
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
import pandas as pd
import math
import random
from tqdm import tqdm
from scipy.stats import entropy as kl_div
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from inf_density import InformationDensity
from sklearn.neighbors import KNeighborsClassifier
from egl import expected_gradient, Softmax
from mval_sup import MVAL_SUP

# Modifies the boolean list output from the VPLNN in order to meet sanity checks:
# First, that the model we are training has high confidence in its label.
# Second, that the label and vector meet the smoothness constraint. 
# (closest labeled point has same label)
def enforce_smoothness(model, X_unlabel, y_unlabel_pred, X_label, y_label, pseudo_bool_lst):
	kNC = KNeighborsClassifier(n_neighbors=X_label.shape[0], weights="distance", n_jobs=-1)
	kNC.fit(X_label, y_label)
	y_pred = kNC.predict(X_unlabel)
	model_conf = (np.amax(model.predict_proba(X_unlabel), axis=1) >= 0.99)
	pseudo_bool_lst = np.logical_and(pseudo_bool_lst, model_conf)
	return np.logical_and(pseudo_bool_lst, np.equal(y_unlabel_pred, y_pred))

# Function allows the merging on vectors and labels from two distinct sources.
def merge_sets(X_pseudo, y_pseudo, X_label, y_label):
	if X_pseudo is None:
		return X_label, y_label
	else:
		X_fit = np.vstack((X_label, X_pseudo))
		y_fit = np.append(y_label, y_pseudo)
		return X_fit, y_fit

# Transfers vectors and labels from source to traget based on the list indices.
def migrate_vectors(X_from, y_from, X_to, y_to, indices):
	indices = np.sort(indices)[::-1]
	for index in indices:
		curRow = X_from[index]
		curLabel = y_from[index]

		#Add values
		if X_to is None:
			X_to = np.array([curRow])
			y_to = np.array([curLabel])
		else:
			X_to = np.vstack((X_to, curRow))
			y_to = np.append(y_to, [curLabel])

		#Remove values
		X_from = np.delete(X_from, index, 0)
		y_from = np.delete(y_from, index)

	return X_from, y_from, X_to, y_to

# Compiles all heuristics to be used by both VQNN and VPLNN in an easy to use manner.
def accumulate_heuristics(model, X_label, y_label, X_unlabel, y_unlabel, Cid, rank_conv = True):

	#Boundary-Based Lists
	LinearBoundLst = get_bound_list(X_label, X_unlabel, y_label, y_unlabel, kernel="linear")
	RBFBoundLst = get_bound_list(X_label, X_unlabel, y_label, y_unlabel, kernel="rbf")
	PolyBoundLst = get_bound_list(X_label, X_unlabel, y_label, y_unlabel, kernel="poly")
	SigmoidBoundLst = get_bound_list(X_label, X_unlabel, y_label, y_unlabel, kernel="sigmoid")

	#Angle
	AngleLst = get_angle(X_unlabel, X_label)

	#Query-By-Committe
	qbcLst = qbc_metric(model, X_label, y_label, X_unlabel)

	#Expected-Gradient-Length
	gradLst = expected_gradient(X_label, y_label, X_unlabel)

	#ChangeDueToPerturbation
	perturbLst = perturbation(model, X_label, y_label, X_unlabel)

	#CoTraining
	coTrainLst = co_training_consensus(model, X_label, y_label, X_unlabel)

	#CoLearning
	coLearnLst = co_learning_consensus(X_label, y_label, X_unlabel)

	EntropyLst = []
	MarginConfidenceLst = []
	RatioConfidenceLst = []
	LeastConfidenceLst = []
	CosDensityLst = []
	EucDensityLst = []

	NumFeatures = X_label.shape[1]
	NumClasses = len(np.unique(y_label))

	ClassLst = []
	FeatureLst = []

	for index, X_val in enumerate(X_unlabel):
		X_cur = X_val.reshape(1, -1)

		probLst = model.predict_proba(X_cur)[0]

		#Pre-sort to speed up Confidence metrics
		probLst[::-1].sort()
		
		#Get uncertainty metrics
		EntropyLst.append(entropy_score(probLst))
		MarginConfidenceLst.append(margin_confidence(probLst, sorted=True))
		RatioConfidenceLst.append(ratio_confidence(probLst, sorted=True))
		LeastConfidenceLst.append(least_confidence(probLst, sorted=True))

		CosDensity, EucDensity = Cid.density(X_cur)

		CosDensityLst.append(CosDensity)
		EucDensityLst.append(EucDensity)

		ClassLst.append(NumClasses)
		FeatureLst.append(NumFeatures)

	EntropyLst = EntropyLst
	MarginConfidenceLst = MarginConfidenceLst
	RatioConfidenceLst = RatioConfidenceLst
	LeastConfidenceLst = LeastConfidenceLst

	if rank_conv:
		perturbLst = conv_to_rank(perturbLst)
		LinearBoundLst = conv_to_rank(LinearBoundLst)
		RBFBoundLst = conv_to_rank(RBFBoundLst)
		PolyBoundLst = conv_to_rank(PolyBoundLst)
		SigmoidBoundLst = conv_to_rank(SigmoidBoundLst)
		gradLst = conv_to_rank(gradLst)
		AngleLst = conv_to_rank(AngleLst)
		CosDensityLst = conv_to_rank(CosDensityLst)
		EucDensityLst = conv_to_rank(EucDensityLst)
		EntropyLst = conv_to_rank(EntropyLst)
		MarginConfidenceLst = conv_to_rank(MarginConfidenceLst)
		RatioConfidenceLst = conv_to_rank(RatioConfidenceLst)
		LeastConfidenceLst = conv_to_rank(LeastConfidenceLst)

	HeuristicMatrix = np.column_stack((
										coLearnLst,
										coTrainLst,
										perturbLst,
										LinearBoundLst,
										RBFBoundLst,
										PolyBoundLst,
										SigmoidBoundLst,
										gradLst,
										qbcLst,
										AngleLst,
										CosDensityLst,
										EucDensityLst,
										EntropyLst,
										MarginConfidenceLst,
										RatioConfidenceLst,
										LeastConfidenceLst,
										ClassLst,
										FeatureLst
									))
	return HeuristicMatrix

# Scales value ssuch that largest is 1 and smallest is 0.
def conv_to_rank(inLst):
	maxV = max(inLst)
	minV = min(inLst)
	diffV = maxV - minV
	if diffV == 0:
		return inLst
	return [(perf - minV)/diffV for perf in inLst]

# Calculates consensus for our Co-learning metric.
def co_learning_consensus(X_label, y_label, X_unlabel):
	models = [
		SVC(),
		MLPClassifier(),
		RandomForestClassifier(),
		LogisticRegression()
	]
	firstPred = None
	boolLst = np.array([True for i in range(len(X_unlabel))])
	for model in models:
		model.fit(X_label, y_label)
		cur_pred = model.predict(X_unlabel)
		if firstPred is None:
			firstPred = cur_pred
			continue
		boolLst = np.logical_and(cur_pred == firstPred, boolLst)
	return boolLst.astype(int)

# Calculates consensus for our Co-training metric.
def co_training_consensus(model, X_label, y_label, X_unlabel):
	#First, split feature set.
	N_FEATURES = X_label.shape[1]
	XL1 = X_label[:,:N_FEATURES // 2]
	XL2 = X_label[:, N_FEATURES // 2:]
	XU1 = X_unlabel[:,:N_FEATURES // 2]
	XU2 = X_unlabel[:, N_FEATURES // 2:]
	#Then train
	model1 = clone(model)
	model1.fit(XL1, y_label)
	model2 = clone(model)
	model2.fit(XL2, y_label)
	#Get preds
	preds1 = model1.predict(XU1)
	preds2 = model2.predict(XU2)
	#Return 1 if same, 0 otherwise
	return (preds1 == preds2).astype(int)

# Calculates our perturbation metric.
def perturbation(model, X_label, y_label, X_unlabel):
	model.fit(X_label, y_label)
	orig_probs = model.predict_proba(X_unlabel)
	perturb_X = X_unlabel + np.random.normal(0, 1, X_unlabel.shape)
	perturb_probs = model.predict_proba(perturb_X)
	perturbLst = kl_div(orig_probs.T, qk=perturb_probs.T)
	return perturbLst/max(perturbLst)

# Converts model performance scores for labeling a given unlabeled vector into a ranking from 1 (best) to 0.

def perf_inc_rank(model, X_label, y_label, X_unlabel, y_unlabel, X_test, y_test):
	perfLst = []
	for index, X_val in tqdm(enumerate(X_unlabel)):
		X_cur = X_val.reshape(1, -1)
		y_cur = y_unlabel[index]
		X_new = np.vstack((X_label, X_cur))
		y_new = np.append(y_label, y_cur)
		X_new_unlabel = np.delete(X_unlabel, index, 0)
		y_new_unlabel = np.delete(y_unlabel, index)

		SLIDING_SIZE = 5

		scoreLst = []

		cur_perm = np.random.permutation(len(X_new_unlabel))

		for i in range(len(X_new_unlabel) - SLIDING_SIZE + 1):
			X_fit, y_fit = merge_sets(X_new_unlabel[cur_perm][i:(i+SLIDING_SIZE)], y_new_unlabel[cur_perm][i:(i+SLIDING_SIZE)], X_new, y_new)
			model.fit(X_fit, y_fit)
			scoreLst.append(model.score(X_test, y_test))

		perfLst.append(np.mean(scoreLst))
	return conv_to_rank(perfLst)

# Calculates consensus variance for Query-by-committee.
# Not currently used.
def qbc_var(model, X_label, y_label, X_unlabel, numModels = 100):
	outputLst = []
	for i in range(numModels):
		perm = np.random.permutation(len(X_label))
		curModel = clone(model)
		curModel.fit(X_label[perm], y_label[perm])
		outputLst.append(curModel.predict(X_unlabel))
	return np.var(np.transpose(outputLst), axis=0)

# Calculates consensus for our Query-by-committee metric.
def qbc_metric(model, X_label, y_label, X_unlabel, numModels = 100):
	outputLst = []
	for i in range(numModels):
		perm = np.random.permutation(len(X_label))
		curModel = clone(model)
		curModel.fit(X_label[perm], y_label[perm])
		outputLst.append(curModel.predict(X_unlabel))
	outputLst = np.transpose(outputLst)
	consensusLst = []
	for i in range(len(outputLst)):
		consensusLst.append(100*max(np.bincount(outputLst[i]))/numModels)
	return consensusLst

# Function that handles data loading and feature manipulation in one spot, leading to a simple API call.
def import_data_dataset(dataFile, dataColumns=None, catColumns = [], dropColumns=[], className='Class', sep=',', pcaDim = None):
	if dataColumns is not None:
		dataset = pd.read_csv(dataFile, names=dataColumns, sep=sep)
	else:
		dataset = pd.read_csv(dataFile, sep=sep)
	dataset = dataset.dropna()
	for column in catColumns:
		dataset[column] = LabelEncoder().fit_transform(dataset[column])
	for column in dropColumns:
		dataset = dataset.drop(column, axis=1)
	dataset = dataset.apply(pd.to_numeric)
	y = np.array(dataset[className])
	X = dataset.drop(className, axis=1)
	if pcaDim is not None and X.shape[1] > pcaDim:
		#For performance reasons, restrict number of features to pcaDim
		X = PCA(n_components=pcaDim).fit_transform(X)
	return np.array(X), np.array(y)

#Init seed split for DLAL initialization
def init_split(X, y, randomSplit=False, init_size=0.1, stratify=True, add_medoids=False):
	if randomSplit:
		NumClasses = len(np.unique(y))
		N = len(X)
		init_size = max(init_size, (1.0 + NumClasses)/N)
		if stratify:
			X_rest, X_init, y_rest, y_init = train_test_split(X, y, test_size=init_size, stratify=y)
		else:
			X_rest, X_init, y_rest, y_init = train_test_split(X, y, test_size=init_size)
	else:
		X_init, y_init = get_repr_pts(X, y, add_medoids=add_medoids)
		X_rest, y_rest = prune_vals(X_init, y_init, X, y)
	return X_rest, X_init, y_rest, y_init

# Prune newly "labeled" vector(s) from unlabled list.
# Except for when called by init split, len(X_label) is expected to be 1.
def prune_vals(X_label, y_label, X_unlabel, y_unlabel):
	for index, X_val in enumerate(X_label):
		boolList = [not (X_unlabel[i] == X_val).all() for i in range(len(X_unlabel))]
		X_unlabel = X_unlabel[boolList]
		y_unlabel = y_unlabel[boolList]
	return X_unlabel, y_unlabel

# Obtain a single point per class.
# Currently not used.
def get_one_pt_per_class(X, y):
	#Shuffle the the points in unision
	shufflePerm = np.random.permutation(len(X))
	X = X[shufflePerm]
	y = y[shufflePerm]

	X_out = None
	y_out = None

	for y_class in np.unique(y):
		X_val = X[y == y_class][0]

		#Add values
		if X_out is None:
			X_out = np.array([X_val])
			y_out = np.array([y_class])
		else:
			X_out = np.vstack((X_out, X_val))
			y_out = np.append(y_out, [y_class])

	return X_out, y_out

#Get center of centroids for init seeds
def get_repr_pts(X, y, tsne=True, add_medoids=False):
	if not add_medoids:
		return get_one_pt_per_class(X, y)
	#https://towardsdatascience.com/how-to-tune-hyperparameters-of-tsne-7c0596a18868
	N = len(X)
	perplexity = math.ceil(N**0.5)

	if tsne:
		X_embedded = TSNE(n_components=2, n_jobs=-1, perplexity=perplexity).fit_transform(X)
	else:
		X_embedded = X

	boolLst = []

	NumClasses = len(np.unique(y))

	maxComp = min(math.ceil(NumClasses*1.5), len(X))

	"""
	# Add K-Medoids
	#pip install scikit-learn-extra
	#from sklearn_extra.cluster import KMedoids

	for metric in ['euclidean', 'cosine', 'manhattan']:

		optModel = None
		optScore = 0.0

		#Have multiple starts to obtain optimal config
		for n_comp in range(2, maxComp):
			model = KMedoids(n_clusters=n_comp, metric=metric, init='k-medoids++')
			model.fit(X_embedded)

			score = silhouette_score(X_embedded, model.predict(X_embedded))

			if score >= optScore:
				optScore = score
				optModel = model

		boolLst.extend([np.all(X_embedded==row,axis=1) for row in optModel.cluster_centers_])
	"""

	# Then GMM:
	optModel = None
	optScore = 0.0

	#Have multiple starts to obtain optimal config
	for n_comp in range(2, maxComp):
		model = mixture.GaussianMixture(n_components=n_comp, covariance_type='full').fit(X)
		model.fit(X_embedded)

		score = silhouette_score(X_embedded, model.predict(X_embedded))

		"""
		If a score is equivalent with a larger number of components,
		we choose the larger one in order to maximize the number of
		"""
		if score >= optScore:
			optScore = score
			optModel = model

	centers = np.empty(shape=(optModel.n_components, X_embedded.shape[1]))
	for i in range(optModel.n_components):
		density = scipy.stats.multivariate_normal(cov=optModel.covariances_[i], mean=optModel.means_[i]).logpdf(X_embedded)
		centers[i, :] = X_embedded[np.argmax(density)]

	boolLst.extend([np.all(X_embedded==row,axis=1) for row in centers])

	boolLst = np.any(boolLst, axis=0)

	centers = X[boolLst]
	y_center = y[boolLst]

	#Finally, add a random vector from each class.
	#This simulates the oracle noticing the returned clusters do not represent all classes and picking a random vector for the missing classes.
	while len(np.unique(y_center)) != len(np.unique(y)):
		indexChoice = random.choice([i for i, x in enumerate(~np.isin(y, y_center)) if x])
		boolLst[indexChoice] = True
		centers = X[boolLst]
		y_center = y[boolLst]

	return centers, y_center

# Calculate the orthogonality metric.
def get_angle(X_unlabel, X_label):
	simMatrix = cosine_similarity(X_unlabel, X_label)
	return np.amin(simMatrix, axis=1)

# Calculate boundary ratios for all unlabeled for given the current kernel.
def get_bound_list(X_label, X_unlabel, y_label, y_unlabel, kernel='linear'):
	model = SVC(kernel=kernel, gamma='scale', break_ties=True)
	model.fit(X_label, y_label)
	avg_bound = avg_boundary_dist(model, X_label)
	boundLst = dist_from_decision_boundary(model, X_unlabel)/avg_bound
	return boundLst

# Helper function for get_bound_list
def avg_boundary_dist(model, X):
	"""
	For linear kernel, the decision boundary is y = w * x + b, 
	the distance from point x to the decision boundary is y/||w||.
	"""

	dist = dist_from_decision_boundary(model, X)

	return np.mean(dist)

# Helper function for get_bound_list
def dist_from_decision_boundary(model, x):
	"""
	For linear kernel, the decision boundary is y = w * x + b, 
	the distance from point x to the decision boundary is y/||w||.

	Return the absolute value of the SMALLEST distance the point is to any decision boundary.
	"""
	y = np.abs(model.decision_function(x))

	try:
		#If multiclass, return the SMALLEST distance the point is to any decision boundary.
		y = np.min(y, axis=1)
	except:
		pass

	try:
		w_norm = np.linalg.norm(model.coef_)
		dist = y / w_norm
	except:
		dist = y

	return dist

# https://github.com/rmunro/uncertainty_sampling_numpy/blob/master/uncertainty_sampling_numpy.py
def margin_confidence(prob_dist, sorted=False):
	""" Margin of Confidence Uncertainty Sampling
	Returns the uncertainty score of a probability distribution using
	margin of confidence sampling in a 0-1 range where 1 is the most uncertain
	
	Assumes probability distribution is a numpy 1d array like: 
		[0.0321, 0.6439, 0.0871, 0.2369]
		
	Keyword arguments:
		prob_dist -- a numpy array of real numbers between 0 and 1 that total to 1.0
		sorted -- if the probability distribution is pre-sorted from largest to smallest
	"""
	if not sorted:
		prob_dist[::-1].sort() # sort probs so that largest is at prob_dist[0]		
		
	difference = (prob_dist[0] - prob_dist[1])
	margin_conf = 1 - difference 
	
	return margin_conf	

def ratio_confidence(prob_dist, sorted=False):
	"""Ratio of Confidence Uncertainty Sampling 
 
	Returns the uncertainty score of a probability distribution using
	ratio of confidence sampling in a 0-1 range where 1 is the most uncertain
	
	Assumes probability distribution is a numpy 1d array like: 
		[0.0321, 0.6439, 0.0871, 0.2369]
		
	Keyword arguments:
		prob_dist -- a numpy array of real numbers between 0 and 1 that total to 1.0
		sorted -- if the probability distribution is pre-sorted from largest to smallest
	"""
	if not sorted:
		prob_dist[::-1].sort() # sort probs so that largest is at prob_dist[0]		
		
	ratio_conf = prob_dist[1] / prob_dist[0]
	
	return ratio_conf

def least_confidence(prob_dist, sorted=False):
	""" Least Confidence Uncertainty Sampling 
	Returns the uncertainty score of a probability distribution using
	least confidence sampling in a 0-1 range where 1 is the most uncertain
	
	Assumes probability distribution is a numpy 1d array like: 
		[0.0321, 0.6439, 0.0871, 0.2369]
		
	Keyword arguments:
		prob_dist -- a numpy array of real numbers between 0 and 1 that total to 1.0
		sorted -- if the probability distribution is pre-sorted from largest to smallest
	"""
	if sorted:
		simple_least_conf = prob_dist[0] # most confident prediction
	else:
		simple_least_conf = np.nanmax(prob_dist) # most confident prediction, ignoring NaNs
				
	num_labels = float(prob_dist.size) # number of labels
	
	normalized_least_conf = (1 - simple_least_conf) * (num_labels / (num_labels -1))
	
	return normalized_least_conf

def entropy_score(prob_dist):
	""" Entropy-Based Uncertainty Sampling 
	Returns the uncertainty score of a probability distribution using
	entropy score
	
	Assumes probability distribution is a numpy 1d array like: 
		[0.0321, 0.6439, 0.0871, 0.2369]
		
	Keyword arguments:
		prob_dist -- a numpy array of real numbers between 0 and 1 that total to 1.0
		sorted -- if the probability distribution is pre-sorted from largest to smallest
	"""
	log_probs = prob_dist * np.log2(prob_dist + 0.00001) # multiply each probability by its base 2 log
	raw_entropy = 0-np.sum(log_probs)

	normalized_entropy = raw_entropy / np.log2(prob_dist.size)
	
	return normalized_entropy
