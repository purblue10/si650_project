import csv
import re 
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
from sklearn import svm
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.metrics import f1_score

from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer


def pca(train_X, test_X, n):
	start_time = time.time()
	pca = PCA(n_components=n)
	pca.fit(train_X.toarray())
	train_X_pca = pca.transform(train_X.toarray())
	test_X_pca = pca.transform(test_X.toarray())
	print("--- %s seconds ---" % (time.time() - start_time))
	return pca, train_X_pca, test_X_pca

def rpca(train_X, test_X, n):
	start_time = time.time()
	pca = RandomizedPCA(n_components=n)
	pca.fit(train_X.toarray())
	train_X_pca = pca.transform(train_X.toarray())
	test_X_pca = pca.transform(test_X.toarray())
	print("--- %s seconds ---" % (time.time() - start_time))
	return pca, train_X_pca, test_X_pca

# dimension reduction
# start_time = time.time()
# pca = PCA(n_components=1000)
# pca.fit(train_X.toarray())
# train_X_pca = pca.transform(train_X.toarray())
# test_X_pca = pca.transform(test_X.toarray())
# print("--- %s seconds ---" % (time.time() - start_time))


def chisq(train_X, train_y, test_X, kN):
	start_time = time.time()
	ch2 = SelectKBest(chi2, k = kN)
	ch2.fit(train_X, train_y)
	train_X_ch2 = ch2.transform(train_X)
	test_X_ch2 = ch2.transform(test_X)
	print("--- %s seconds ---" % (time.time() - start_time))
	return ch2, train_X_ch2, test_X_ch2