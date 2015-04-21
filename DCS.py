from scipy import sparse
import numpy as np
import csv
import re 
import misc
import time
import fs
import data_process as dp

from brew.selection.dynamic.ola import OLA
from brew.selection.dynamic.ola import OLA2
from brew.selection.dynamic.lca import LCA
from brew.selection.dynamic.lca import LCA2
from brew.selection.dynamic.knora import *
from brew.selection.dynamic.probabilistic import *

from brew.generation.bagging import Bagging
from brew.base import EnsembleClassifier

from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import RandomizedPCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
from sklearn import svm
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import chi2, SelectKBest
from sklearn import cross_validation
from sklearn.metrics import f1_score
from sklearn.ensemble import AdaBoostClassifier
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.neighbors import KDTree

class dynamicClassifierSelection:
	def __init__(self, train_X, test_X, train_y, test_y, classifiers):
		self.train_X = train_X
		self.test_X = test_X
		self.train_y = train_y
		self.test_y = test_y
		self.classifiers = classifiers
		self.knn = {}
		self.knn2 = {}
		self.CLFs = {}

	def setClassifiers(self, classifiers):
		self.classifiers = classifiers
		self.CLFs = {}

	def splitData(self, cvn = 5, test_size_value = 0.2):
		sss = StratifiedShuffleSplit(self.train_y, n_iter=cvn, test_size=test_size_value)
		fold_tr = [] 		#(train_X1_tr, train_X2_tr, train_X3_tr)
		fold_val = []	#(train_X1_val, train_X2_val, train_X2_val)
		fold_total = []	#{'tr':train_Xtot_tr, 'val':train_Xtot_val}
		fold_y = [] 		# {'tr':train_y_tr, 'val':train_y_val}
		for train_index, test_index in sss:
			x1_tr, x1_val = self.train_X[0][train_index], self.train_X[0][test_index]
			x2_tr, x2_val = self.train_X[1][train_index], self.train_X[1][test_index]
			x3_tr, x3_val = self.train_X[2][train_index], self.train_X[2][test_index]
			fold_tr.append((x1_tr, x2_tr, x3_tr))
			fold_val.append((x1_val, x2_val, x3_val))

			xTot_tr, xTot_val = self.train_X[3][train_index], self.train_X[3][test_index]
			fold_total.append({'tr':xTot_tr, 'val':xTot_val})

			train_y_tr, train_y_val = self.train_y[train_index], self.train_y[test_index]
			fold_y.append({'tr':train_y_tr, 'val':train_y_val})
		self.fold_tr = fold_tr
		self.fold_val = fold_val
		self.fold_total = fold_total
		self.fold_y = fold_y

	def chooseFold(self, n):
		if n not in self.knn:
			start_time = time.time()
			nbrs = NearestNeighbors(n_neighbors=50, algorithm='ball_tree').fit(self.fold_total[n]['val'])
			neighbors = nbrs.kneighbors(self.test_X[3], return_distance=True)
			self.knn[n] = {"dist":neighbors[0], "index":neighbors[1]}
			print("KNN-%d: --- %s seconds ---" % (n, (time.time() - start_time)))
		if n not in self.CLFs:
			CLF={}
			for i in range(len(self.classifiers)):
				CLF[i] = self.classifiers[i].fit(self.fold_tr[n][i], self.fold_y[n]['tr'])
			self.CLFs[n] = CLF
		return self.fold_tr[n], self.fold_val[n], self.fold_total[n], self.fold_y[n]

	def chooseFold2(self, n):
		if n not in self.knn2:
			start_time = time.time()
			neighbors = []
			for i in range(len(self.classifiers)):
				print i
				if i!=2:
					nb = NearestNeighbors(n_neighbors=50, algorithm='ball_tree').fit(self.fold_val[n][i])
					neighbor = nb.kneighbors(self.test_X[i], return_distance=True)
				elif i==2:
					nb = NearestNeighbors(n_neighbors=50, algorithm='ball_tree').fit(self.fold_val[n][i].toarray())
					neighbor = nb.kneighbors(self.test_X[i].toarray(), return_distance=True)
				neighbors.append(neighbor)
			# neighbors = []
			# [nbrs[i].kneighbors(self.test_X[i], return_distance=True) if i!=2 else nbrs[i].kneighbors(self.test_X[i].toarray, return_distance=True) for i in range(len(nbrs))]
			self.knn2[n] = [{"dist":neighbors[i][0], "index":neighbors[i][1]} for i in range(len(neighbors))]
			print("KNN2-%d: --- %s seconds ---" % (n, (time.time() - start_time)))
		if n not in self.CLFs:
			CLF={}
			for i in range(len(self.classifiers)):
				CLF[i] = self.classifiers[i].fit(self.fold_tr[n][i], self.fold_y[n]['tr'])
			self.CLFs[n] = CLF
		return self.fold_tr[n], self.fold_val[n], self.fold_total[n], self.fold_y[n]

	def priori_probabilities(self, clf, nn_X, nn_y, distances):
	    # in the A Priori method, the 'x' is not used
		proba = clf.predict_proba(nn_X)
		proba = np.hstack((proba, np.zeros((proba.shape[0],1))))

		d = dict(list(enumerate(clf.classes_)))
		col_idx = np.zeros(nn_y.size,dtype=int)
		for i in range(nn_y.size):
			col_idx[i] = d[nn_y[i]] if nn_y[i] in d else proba.shape[1] - 1
		
		probabilities = proba[np.arange(col_idx.size), col_idx]
		delta = 1./(distances + 10e-8)
		
		p_correct = np.sum(probabilities * delta) / np.sum(delta)
		return p_correct

	def posterior_probabilities(self, clf, nn_X, nn_y, distances, x):
		[w_l] = clf.predict(x)
		[idx_w_l] = np.where(nn_y == w_l)

		# in the A Posteriori method the 'x' is used
		proba = clf.predict_proba(nn_X)
		proba = np.hstack((proba, np.zeros((proba.shape[0],1))))

		# if the classifier never classifies as class w_l, P(w_l|psi_i) = 0
		proba_col = proba.shape[1] - 1
		if w_l in clf.classes_:
		    proba_col = np.where(clf.classes_ == w_l)

		delta = 1./(distances + 10e-8)
		numerator = sum(proba[idx_w_l, proba_col].ravel() * delta[idx_w_l])
		denominator = sum(proba[:, proba_col].ravel() * delta)
		value = float(numerator) / (denominator + 10e-8)
		return value

# K: # of nearest neighbor
	def prob_selection(self, n, estimator, K=5, threshold=0.1, second=False):
		train_X_tr, train_X_val, train_X_total, train_y_tr_val = self.chooseFold(n)
		CLF = self.CLFs[n]

		prediction_output = []
		prediction_prob = []
		for i in range(len(self.test_X[0])):
			W = [ CLF[j].predict(self.test_X[j][i])[0]  for j in range(len(CLF))]
			# if len(set(W))==1:
			# 	prediction_output.append(W[0])

			# 	continue
			
			if second==False:
				idx = self.knn[n]['index'][i][:K]
				distances = self.knn[n]['dist'][i][:K]
			
			idx_selected, prob_selected = [], []
			all_probs = np.zeros(len(CLF))
			scores = []
			for j in CLF:
				if second==True:
					idx = self.knn2[n][j]['index'][i][:K]
					distances = self.knn2[n][j]['dist'][i][:K]
				nn_X = train_X_val[ j ][idx,:]
				nn_y = train_y_tr_val['val'][idx]
				if estimator == "priori":
					prob = self.priori_probabilities(CLF[j], nn_X, nn_y, distances)
				elif estimator == "posterior":
					prob = self.posterior_probabilities(CLF[j], nn_X, nn_y, distances, self.test_X[ j ][i])
				if j==1 and self.test_X[j][i].mean() == 0:
					prob = 0
				if prob > 0.5:
					idx_selected = idx_selected + [j]
  					prob_selected = prob_selected + [prob]
			# 		scores.append(prob)
			# 	else:
			# 		scores.append(0)

			# best = np.argmax(scores)
			# prediction_output.append(CLF[best].predict(self.test_X[best][i])[0])

				all_probs[j] = prob

			if len(prob_selected) == 0:
				prob_selected = [np.max(all_probs)]
				idx_selected = [np.argmax(all_probs)]

			p_correct_m = max(prob_selected)
			m = np.argmax(prob_selected)

			selected = True
			diffs = []
			for j, p_correct_j in enumerate(prob_selected):
				d = p_correct_m - p_correct_j
				diffs.append(d)
				if j != m and d < threshold:
					selected = False

			if selected:
				selected_classifier = CLF[idx_selected[m]]
				best = idx_selected[m]
			else:
				idx_selected = np.asarray(idx_selected)
				mask = np.array(np.array(diffs) < threshold, dtype=bool)
				k = np.random.choice(idx_selected[mask])
				selected_classifier = CLF[k]
				best = k

			prediction_output.append(selected_classifier.predict(self.test_X[best][i])[0])
			prediction_prob.append(selected_classifier.predict_proba(self.test_X[best][i]))
		return prediction_output, prediction_prob

	def oracle(self, n):
		train_X_tr, train_X_val, train_X_total, train_y_tr_val = self.chooseFold(n)
		CLF = self.CLFs[n]
		pred = [ CLF[i].predict(self.test_X[i]) for i in CLF]
		pred_prob = [ CLF[i].predict_proba(self.test_X[i]) for i in CLF]
		result = [ misc.evaluation2([pred[i],pred_prob[i]], self.test_y) for i in range(len(pred))]
		final =[]
		for i in range(len(result[0])):
			tp = (result[0][i][2],result[1][i][2],result[2][i][2])
			if True in tp:
				idx = tp.index(True)
				final.append((result[idx][i][1],True))
			else:
				final.append((result[0][i][1], False))
		numTrue = len([ row for row in final if row[1]==True])
		numFalse = len([ row for row in final if row[1]==False])
		numTrue_zero = len([ row for row in final if row[1]==True and row[0]==0])
		numTrue_one = len([ row for row in final if row[1]==True and row[0]==1])
		n_zero = len([ row for row in final if row[0]==0])
		n_one = len([ row for row in final if row[0]==1])

		print("Total Accuracy: %0.5f " % (numTrue/float(len(final))))
		print("0-label Accuracy: %0.5f (%d / %d)" % (numTrue_zero/float(n_zero), numTrue_zero, n_zero))
		print("1-label Accuracy: %0.5f (%d / %d)" % (numTrue_one/float(n_one), numTrue_one, n_one))
		print "Precision : %f,  Recall %f" % (numTrue_one/float(n_one), numTrue_one/float(numTrue_one+n_zero-numTrue_zero))

	def OLA(self, n, K = 5, second=False):
		train_X_tr, train_X_val, train_X_total, train_y_tr_val = self.chooseFold(n)
		CLF = self.CLFs[n]

		prediction_output = []
		prediction_prob = []
		for i in range(len(self.test_X[0])):
			if second==True:
				score=[CLF[ j ].score(train_X_val[ j ][self.knn2[n][j]['index'][i][:K],:], train_y_tr_val['val'][self.knn2[n][j]['index'][i][:K]] ) for j in range(len(CLF))]
				# idx = 
			else:
				idx = self.knn[n]['index'][i][:K]
				score=[CLF[ j ].score(train_X_val[ j ][idx,:], train_y_tr_val['val'][idx] ) for j in range(len(CLF))]
			if self.test_X[1][i].mean() == 0:
				score[1] =0
			best_clf = np.argmax(score)
			prediction_output.append( CLF[best_clf].predict(self.test_X[best_clf][i])[0])
			prediction_prob.append( CLF[best_clf].predict_proba(self.test_X[best_clf][i])[0])
		return prediction_output, prediction_prob
	def LCA(self, n, K = 5, second=False):
		train_X_tr, train_X_val, train_X_total, train_y_tr_val = self.chooseFold(n)
		CLF = self.CLFs[n]

		prediction_output = []
		prediction_prob = []
		for i in range(len(self.test_X[0])):
			if second==True:
				# idx = self.knn2[n][j]['index'][i][:K]
				pred = [CLF[ j ].predict(train_X_val[ j ][self.knn2[n][j]['index'][i][:K],:]) for j in range(len(CLF))]
			else:
				idx = self.knn[n]['index'][i][:K]
				pred = [CLF[ j ].predict(train_X_val[ j ][idx,:]) for j in range(len(CLF))]
			w = [CLF[ j ].predict(self.test_X[ j ][i])[0] for j in range(len(CLF))]
			scores = []
			for k in range(len(CLF)):
				if second==True:
					y = train_y_tr_val['val'][self.knn2[n][ k ]['index'][i][:K]]
				else:
					y = train_y_tr_val['val'][idx]
				pp = len([ pred[k][p] for p in range(len(pred[k])) if pred[k][p]==w[k] and pred[k][p]==y[p]])
				ip = len([ pred[k][p] for p in range(len(pred[k])) if pred[k][p]==w[k]])
				if ip!=0:
					scores.append(pp/float(ip))
				else:
					scores.append(0)
			if self.test_X[1][i].mean() == 0:
				scores[1] =0
			best_clf = np.argmax(scores)
			# prediction2.append(w[best_clf])
			prediction_output.append( CLF[best_clf].predict(self.test_X[best_clf][i])[0])
			prediction_prob.append( CLF[best_clf].predict_proba(self.test_X[best_clf][i])[0])
		return prediction_output, prediction_prob

	# 			if prob < 0.5:
	# 				scores.append(-1)
	# 			else:
	# 				scores.append(prob)

	# 			if test_X[1][i].mean() == 0:
	# 				scores[1] = -1
				
	# 			best = np.argmax(scores)
	# prediction_prior.append(CLFs[best].predict(test_X[best][i])[0])


# a = posterior_probabilities(CLFs[k], nn_X, nn_y, distances, test_X[k][i])


# fold_tr, fold_val, fold_total, fold_y = splitData(train_X, train_y)
# def splitData(train_X, train_y):
# 	sss = StratifiedShuffleSplit(train_y, 5, test_size=0.2)
# 	fold_tr = [] 	#(train_X1_tr, train_X2_tr, train_X3_tr)
# 	fold_val = []	#(train_X1_val, train_X2_val, train_X2_val)
# 	fold_total = []	#(train_Xtot_tr, train_Xtot_val)
# 	fold_y = [] 	#(train_y_tr, train_y_val)
# 	# fold : train_X1_tr, train_X1_val, train_X2_tr, train_X2_val, train_X3_tr, train_X3_val
# 	for train_index, test_index in sss:
# 		x1_tr, x1_val = train_X[0][train_index], train_X[0][test_index]
# 		x2_tr, x2_val = train_X[1][train_index], train_X[1][test_index]
# 		x3_tr, x3_val = train_X[2][train_index], train_X[2][test_index]
# 		fold_tr.append((x1_tr, x2_tr, x3_tr))
# 		fold_val.append((x1_val, x2_val, x3_val))
# 		xTot_tr, xTot_val = train_X[3][train_index], train_X[3][test_index]
# 		fold_total.append((xTot_tr, xTot_val))
# 		train_y_tr, train_y_val = train_y[train_index], train_y[test_index]
# 		fold_y.append((train_y_tr, train_y_val))
# 	return fold_tr, fold_val, fold_total, fold_y


# # def dynamicClassifierSelection(CLFs, train_X, test_X, train_y, test_y):




# def selection(CLFs, train_X_tr, train_X_val, train_X_total, train_y_tr_val, test_X, test_y):

# clf1 = LogisticRegression(penalty="l2", dual=False, C=1)
# clf2 = LogisticRegression(penalty="l2", dual=True, C=1)
# clf3 = LogisticRegression(penalty="l2", dual=True, C=5)
# CLFs=[clf1, clf2, clf3]


# train_X_tr = fold_tr[0]
# train_X_val = fold_val[0]
# train_X_total = fold_total[0]
# train_y_tr_val = fold_y[0]



# for i in range(len(CLFs)):
# 	CLFs[i] = CLFs[i].fit(train_X_tr[i], train_y_tr_val[0])

# s = time.time()

# print(time.time() - s)

# nbrs = []
# for i in range(len(CLFs)):
# 	if i!=2:
# 		nb = NearestNeighbors(n_neighbors=15, algorithm='ball_tree').fit(train_X_val[i])
# 	if i==2:
# 		nb = NearestNeighbors(n_neighbors=15, algorithm='ball_tree').fit(train_X_val[i].toarray())
# 	nbrs.append(nb)

# test_X[2] = test_X[2].toarray()
# indices = [nbrs[i].kneighbors(test_X[i], return_distance=True) for i in range(len(nbrs))]
# # ola
# prediction1 = []
# for i in range(len(indices[0][1])):
# 	# idx = indices[i]
# 	score=[CLFs[n].score(train_X_val[n][indices[n][1][i],:], train_y_tr_val[1][indices[n][1][i]]) for n in range(len(CLFs))]
# 	if test_X[1][i].mean() == 0:
# 		score[1] =0
# 	best_clf = np.argmax(score)
# 	prediction1.append( CLFs[best_clf].predict(test_X[best_clf][i])[0])

# #lca
# prediction2 = []
# for i in range(len(indices)):
# 	idx = indices[i]
# 	w = [CLFs[n].predict(test_X[n][i])[0] for n in range(len(CLFs))]
# 	pred = [CLFs[n].predict(train_X_val[n][idx,:]) for n in range(len(CLFs))]
# 	scores = []
# 	for n in range(len(CLFs)):
# 		y = train_y_tr_val[1][idx]
# 		pp = len([ pred[n][p] for p in range(len(pred[n])) if pred[n][p]==w[n] and pred[n][p]==y[p]])
# 		ip = len([ pred[n][p] for p in range(len(pred[n])) if pred[n][p]==w[n]])
# 		if ip!=0:
# 			scores.append(pp/float(ip))
# 		else:
# 			scores.append(0)
# 	if test_X[1][i].mean() == 0:
# 		score[1] =0
# 	best_clf = np.argmax(score)
# 	prediction2.append(w[best_clf])


# ## a prior

# train_y_tr_val[1][idx]

# prediction_prior = []
# for i in range(len(indices[0])):
# 	idx = indices[1][i][:10]
# 	distances = indices[0][i][:10]
# 	score = []
# 	for k in range(len(CLFs)):
# 		nn_X = train_X_val[k][idx,:]
# 		nn_y = train_y_tr_val[1][idx]
# 		a = priori_probabilities(CLFs[k], nn_X, nn_y, distances)
# 		score.append(a)
# 	if test_X[1][i].mean() == 0:
# 		score[1] =0
# 	best = np.argmax(score)
# 	prediction_prior.append(CLFs[best].predict(test_X[best][i])[0])


# r3 = misc.evaluation(prediction_prior,test_y)

# ## posterior
# prediction_post = []
# for i in range(len(indices[0])):
# 	idx = indices[1][i][:30]
# 	distances = indices[0][i][:30]
# 	score = []
# 	for k in range(len(CLFs)):
# 		nn_X = train_X_val[k][idx,:]
# 		nn_y = train_y_tr_val[1][idx]
# 		a = posterior_probabilities(CLFs[k], nn_X, nn_y, distances, test_X[k][i])
# 		score.append(a)
# 	if test_X[1][i].mean() == 0:
# 		score[1] =0
# 	best = np.argmax(score)
# 	prediction_post.append(CLFs[best].predict(test_X[best][i])[0])

# r4 = misc.evaluation(prediction_post,test_y)

# test_y[100]
# clf = CLFs[0].fit(train_X_tr[i],train_y_tr_val[0])

# def priori_probabilities(clf, nn_X, nn_y, distances):
#     # in the A Priori method, the 'x' is not used
# 	proba = clf.predict_proba(nn_X)
# 	proba = np.hstack((proba, np.zeros((proba.shape[0],1))))
# 	d = dict(list(enumerate(clf.classes_)))
# 	col_idx = np.zeros(nn_y.size,dtype=int)
# 	for i in range(nn_y.size):
# 		col_idx[i] = d[nn_y[i]] if nn_y[i] in d else proba.shape[1] - 1
# 	probabilities = proba[np.arange(col_idx.size), col_idx]
# 	delta = 1./(distances + 10e-8)
# 	p_correct = np.sum(probabilities * delta) / np.sum(delta)
# 	return p_correct

# k=2
# idx = indices[1][100][:10]
# distances = indices[0][100][:10]
# nn_X = train_X_val[k][idx,:]
# nn_y = train_y_tr_val[1][idx]
# a = posterior_probabilities(CLFs[k], nn_X, nn_y, distances, test_X[k][100])

# def posterior_probabilities(clf, nn_X, nn_y, distances, x):
#     [w_l] = clf.predict(x)
#     [idx_w_l] = np.where(nn_y == w_l)
#     # in the A Posteriori method the 'x' is used
#     proba = clf.predict_proba(nn_X)
#     proba = np.hstack((proba, np.zeros((proba.shape[0],1))))
#     # if the classifier never classifies as class w_l, P(w_l|psi_i) = 0
#     proba_col = proba.shape[1] - 1
#     if w_l in clf.classes_:
#         proba_col = np.where(clf.classes_ == w_l)
#     delta = 1./(distances + 10e-8)
#     numerator = sum(proba[idx_w_l, proba_col].ravel() * delta[idx_w_l])
#     denominator = sum(proba[:, proba_col].ravel() * delta)
#     value = float(numerator) / (denominator + 10e-8)
#     return value

# CLFs[0].score(test_X[0],test_y)
# CLFs[1].score(test_X[1],test_y)
# CLFs[2].score(test_X[2],test_y)
# r1 = misc.evaluation(prediction1,test_y)
# r2 = misc.evaluation(prediction2,test_y)
# r3 = misc.evaluation(prediction_prior,test_y)

# def oracle(CLFs, test_X, test_y):
# 	predict1 = logit1.predict(test_X1)
# 	predict2 = logit2.predict(test_X2)
# 	predict3 = logit3.predict(test_X3_ch2)

# result1 = misc.evaluation(predict1.tolist(), test_y)
# result2 = misc.evaluation(predict2.tolist(), test_y)
# result3 = misc.evaluation(predict3.tolist(), test_y)

# # train_X_tr = fold_tr[0]
# # train_X_val = fold_val[0]
# # train_X_total = fold_total[0]

# # s = time.time()# print(time.time() - s)
# # nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(train_X_total[1])
# # print(time.time() - s)

# # kdt = KDTree(train_X_total[1], leaf_size=5000, metric='euclidean')
# # print(time.time() - s)
# # s = time.time()
# # indices = nbrs.kneighbors(test_X[3][:1000], return_distance=False)
# # print(time.time() - s)
# # idx_arr = []
# # i=0
# # for row in test_X[3]:
# # 	i+=1
# # 	idx = kdt.query(row, k=2, return_distance=False)
# # 	idx_arr.append(idx)
# # 	print i

# # print(time.time() - s)
	




