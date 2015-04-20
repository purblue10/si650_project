from scipy import sparse
import numpy as np
import csv
import re 
import misc
import time
import fs
import data_process as dp

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

train_path = "./data/data2/train.json"
test_path = "./data/data2/test.json"


train1, train2, train3 = dp.form_matrix(train_path, type=0)
test1, test2, test3 = dp.form_matrix(test_path, type=0)

# 1. Get matrice
## first classifier
train_X1 = [ row[1][0:1]+row[1][2:]  for row in train1]
train_y = [ row[0]  for row in train1]
test_X1 = [ row[1][0:1]+row[1][2:]  for row in test1]
test_y = [ row[0]  for row in test1]

train_X1 = np.array(train_X1)
test_X1 = np.array(test_X1)
train_y = np.array(train_y)
test_y = np.array(test_y)

## second classifier
train_X2 = []
test_X2 = []
for item in train2:
	dic = {}
	for tag in item[1]:
		dic[tag] = 1 if  tag not in dic else dic[tag]+1
	train_X2.append(dic)

for item in test2:
	dic = {}
	for tag in item[1]:
		dic[tag] = 1 if  tag not in dic else dic[tag]+1
	test_X2.append(dic)

dicVectorizer = DictVectorizer(sparse=False)
train_X2 = dicVectorizer.fit_transform(train_X2)
dicVectorizer = DictVectorizer(sparse=False)
test_X2 = dicVectorizer.fit_transform(test_X2)

## thrid classifier
train_text, train_y3 = misc.getTextAndLabel(train3)
test_text, test_y3 = misc.getTextAndLabel(test3)

vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', lowercase=True, sublinear_tf=True, tokenizer=misc.LemmaTokenizer(), ngram_range=(1,2))

vectorizer.fit(train_text)

train_X3 = vectorizer.transform(train_text)
test_X3 = vectorizer.transform(test_text)

ch2, train_X3_ch2, test_X3_ch2 = fs.chisq(train_X3, train_y3, test_X3, 20000)


train_X1 = sparse.csr_matrix(train_X1)
train_X1 = train_X1.toarray()

train_X_tot = sparse.hstack((sparse.csr_matrix(train_X1), sparse.csr_matrix(train_X2), train_X3_ch2))
test_X_tot = sparse.hstack((sparse.csr_matrix(test_X1), sparse.csr_matrix(test_X2), test_X3_ch2))

#2. Classifier
logit1 = LogisticRegression(penalty="l2", dual=False, C=1)
logit1.fit(train_X1, train_y)

logit2 = LogisticRegression(penalty="l2", dual=True, C=1)
logit2.fit(train_X2, train_y)

logit3 = LogisticRegression(penalty="l2", dual=True, C=5)
logit3.fit(train_X3_ch2, train_y)

#3. prediction
predict1 = logit1.predict(test_X1)
predict2 = logit2.predict(test_X2)
predict3 = logit3.predict(test_X3_ch2)

result1 = misc.evaluation(predict1.tolist(), test_y)
result2 = misc.evaluation(predict2.tolist(), test_y)
result3 = misc.evaluation(predict3.tolist(), test_y)

# Total Accuracy: 0.76086 
# 0-label Accuracy: 0.77592 (4536 / 5846)
# 1-label Accuracy: 0.68458 (790 / 1154)

predict1_prob = logit1.predict_proba(test_X1)
predict2_prob = logit2.predict_proba(test_X2)
predict3_prob = logit3.predict_proba(test_X3_ch2)
r = majority_prob(predict1_prob, predict2_prob, predict3_prob, [0.3, 0.05, 0.65])
result4 = misc.evaluation(r, test_y)

r = majority(result1, result2, result3)
result4 = misc.evaluation(r, test_y)
def majority(result1, result2, result3):
	major = []
	for i in range(len(result1)):
		tp = (result1[i][1], result2[i][1], result3[i][1])
		nTrue = len([ k for k in tp if k==1])
		nFalse = len([ k for k in tp if k==0])
		m = 1 if nTrue>nFalse else 0
		major.append(m)
	return major

def majority_prob(predict1_prob, predict2_prob, predict3_prob, w):
	prob1 = [ row[1] for row in predict1_prob]
	prob2 = [ row[1] for row in predict2_prob]
	prob3 = [ row[1] for row in predict3_prob]
	result = []
	for i in range(len(prob1)):
		p = w[0]*prob1[i] + w[1]*prob2[i] + w[2]*prob3[i]
		label = 1 if p > 0.5 else 0
		result.append(label)
	return result


train_X_tot

score_logit = misc.testCV(logit5, train_X_tot, train_y, 5)

logit5 = LogisticRegression(penalty="l2", dual=False, C=10)
logit5.fit(train_X_tot, train_y)
predict5 = logit5.predict(test_X_tot)
result5 = misc.evaluation(predict5.tolist(), test_y)
# logit_grid = LogisticRegressionCV(Cs=[2000, 2500, 3000, 3500, 4000], cv=5, dual=False, penalty="l2")
# logit_grid.fit(train_X3_ch2, train_y)

# ev = []
# for i in range(len(logit_grid.scores_[1])):
# 	ev.append(np.array([row[i] for row in logit_grid.scores_[1]]).mean())

# logit_grid.C_
# logit_grid.Cs_
# ev




ch2, train_X_ch2, test_X_ch2 = fs.chisq(train_X, train_y, test_X, 5000)

clf = svm.SVC(kernel='rbf', C=500, gamma=0.001, cache_size=500)
score_svm = testCV(clf, train_X, train_y, 5)


ch2, train_X_ch2, test_X_ch2 = fs.chisq(train_X, train_y, test_X, 20000)
logit = LogisticRegression(penalty="l2", dual=True, C=500)
score_logit = testCV(logit, train_X_ch2, train_y, 5)

ch2, train_X_ch2, test_X_ch2 = fs.chisq(train_X, train_y, test_X, 20000)
ada = AdaBoostClassifier(n_estimators=100)
score_ada = testCV(ada, train_X_ch2, train_y, 5)

def testCV(clf, train_X, train_y, cvn):
	start_time = time.time()
	scores = cross_validation.cross_val_score(clf, train_X_ch2, train_y, cv=5)
	mean = "{:.5f}".format(scores.mean())
	sd = "{:.5f}".format(scores.std()*2)
	line = "accuracy: "+ str(mean) +" (+/- " + str(sd) + ")" +", "
	print line
	print("cross validation: --- %s seconds ---" % (time.time() - start_time))
	return scores


# clf3.fit(train_X_ch2, train_y)




train_X1 = [ row[1][0:1]+row[1][2:]  for row in train1[:1000]]
train_y1 = [ row[0]  for row in train1]
test_X1 = [ row[1][0:1]+row[1][2:]  for row in train1]
test_y1 = [ row[0]  for row in test]


train_X1 = np.array(train_X1)
train_y1 = np.array(train_y1)


clf = svm.SVC(kernel='rbf', C=500, gamma=0.001, cache_size=500)
clf = svm.SVC()
clf.fit(train_X1, train_y1)




## Classifier
clf2 = OneVsRestClassifier(svm.SVC(kernel='rbf', C=500, gamma=0.001, cache_size=500))
scores = misc.runCV(clf2, train_X_ch2, train_y, ch2, 5)

clf2.fit(train_X_ch2, train_y)
misc.prediction(clf2, test_X_ch2, ch2)

# prediction
clf = OneVsRestClassifier(svm.SVC(kernel='rbf', C=100, gamma=1))
clf.fit(train_X_ch2, train_y)
misc.prediction(clf, test_X_ch2, ch2)


##############################################################
##############################################################
pca1000, train_X_pca1000, test_X_pca1000 = fs.rpca(train_X, test_X, 1000)
clf = OneVsRestClassifier(svm.SVC(kernel='rbf', C=1, gamma=1, cache_size=500))
clf.fit(train_X_pca1000, train_y)
misc.prediction(clf, test_X_pca1000, pca1000)

## PCA
pca500, train_X_pca500, test_X_pca500 = fs.rpca(train_X, test_X, 500)
pca1000, train_X_pca1000, test_X_pca1000 = fs.rpca(train_X, test_X, 1000)
clf = OneVsRestClassifier(svm.SVC(kernel='rbf', C=500, gamma=0.001, cache_size=500))
clf = OneVsRestClassifier(svm.SVC(kernel='rbf', C=500, gamma=0.1, cache_size=500))
scores2 = misc.runCV(clf, train_X_pca500, train_y, pca500, 5)


pca2000, train_X_pca2000, test_X_pca2000 = fs.pca(train_X, test_X, 2000)
pca5000, train_X_pca5000, test_X_pca5000 = fs.pca(train_X, test_X, 5000)

clf = OneVsRestClassifier(svm.SVC(kernel='rbf', C=500, gamma=0.001))
scores1 = misc.runCV(clf, train_X_pca500, train_y, pca500, 5)
scores2 = misc.runCV(clf, train_X_pca2000, train_y, pca2000, 5)
scores3 = misc.runCV(clf, train_X_pca5000, train_y, pca5000, 5)


pca1000.explained_variance_ratio_



#*** pca test

clf = OneVsRestClassifier(svm.SVC(kernel='rbf', C=0.1, gamma=1, cache_size=500))
scores1 = misc.runCV(clf, train_X_pca500, train_y, pca500, 5)

clf = OneVsRestClassifier(svm.SVC(kernel='rbf', C=1, gamma=1, cache_size=500))
scores2 = misc.runCV(clf, train_X_pca1000, train_y, pca1000, 5)

clf = OneVsRestClassifier(svm.SVC(kernel='rbf', C=10, gamma=0.01, cache_size=500))
scores3 = misc.runCV(clf, train_X_pca500, train_y, pca500, 5)

clf = OneVsRestClassifier(svm.SVC(kernel='rbf', C=500, gamma=0.0001, cache_size=500))
scores4 = misc.runCV(clf, train_X_pca500, train_y, pca500, 5)


clf = OneVsRestClassifier(svm.SVC(kernel='rbf', C=500, gamma=0.001, cache_size=500))
scores11 = misc.runCV(clf, train_X_pca1000, train_y, pca1000, 5)


start_time = time.time()
model = OneVsRestClassifier(svm.SVC(kernel='rbf', cache_size=500))
param_grid = {
    "estimator__C": [0.005, 0.05, 500],
    "estimator__gamma": [0.001, 0.01, 1, 10, 100, 1000]
}
clf_grid = GridSearchCV(model, param_grid=param_grid, score_func=f1_score)
clf_grid.fit(train_X_ch2, train_y)
print("--- %s seconds ---" % (time.time() - start_time))


clf.best_estimator_
clf_grid.best_params_
clf.get_params()

clf2 = OneVsRestClassifier(svm.SVC(kernel='rbf', C=1000, gamma=0.001))

print("--- %s seconds ---" % (time.time() - start_time))
clf2.fit(train_X_ch2, train_y)
prediction = clf2.predict(test_X_ch2)
result = prediction.tolist()
misc.writeResult("./result/submission_ch2_fit_2000_1000_0.0001_lemma.csv", result)

clf2 = OneVsRestClassifier(svm.SVC(kernel='rbf', C=1000, gamma=0.001))
clf2.fit(train_X_pca, train_y)
prediction = clf.predict(test_X_pca)
result = prediction.tolist()
misc.writeResult("./result/submission_pca_fit_2000_1000_0.0001.csv", result)


start_time = time.time()
clf2 = OneVsRestClassifier(svm.SVC(kernel='rbf', C=1000, gamma=0.001))

scores = cross_validation.cross_val_score(clf2, train_X_ch2, train_y, cv=5)
clf = OneVsRestClassifier(svm.SVC(kernel='rbf', C=1000, gamma=0.001))
train_X = train_X_ch2
cvn=5
scores = misc.runCV(clf, train_X_ch2, train_y, ch2, cvn)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("--- %s seconds ---" % (time.time() - start_time))

misc.runCV(clf, train_X_ch2, train_y, ch2)

start_time = time.time()
clf = OneVsRestClassifier(svm.SVC(kernel='rbf'))
scores = cross_validation.cross_val_score(clf, train_X_pca, train_y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("--- %s seconds ---" % (time.time() - start_time))



start_time = time.time()
ch2 = SelectKBest(chi2, k=2000)
ch2.fit(train_X, train_y)
train_X_ch2 = ch2.transform(train_X)
test_X_ch2 = ch2.transform(test_X)


clf2 = OneVsRestClassifier(svm.SVC(kernel='rbf', C=1500, gamma=0.0005))
clf2.fit(train_X_ch2, train_y)
prediction = clf2.predict(test_X_ch2)
result = prediction.tolist()
misc.writeResult("./result/submission_chi2_fit_2000_2000_0.0001.csv", result)
print("--- %s seconds ---" % (time.time() - start_time))

ch2






