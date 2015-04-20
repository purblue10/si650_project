import data_process as dp
import json
import random
import numpy
import time
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


mtx = dp.form_matrix('./data/train.json', type = 2)
X_train = []
y_train = []
for item in mtx:
    dic = {}
    for tag in item[1]:
        if tag not in dic:
            dic[tag] = 1
        else:
            dic[tag] += 1
    X_train.append(dic)
    y_train.append(item[0])
    
v = DictVectorizer(sparse=False)
X_train = v.fit_transform(X_train)


#############SVM : 0.71
#clf = svm.SVC(kernel='linear')

#############MNB : 0.70
#clf = MultinomialNB()

#############BNB : 0.71
#clf = BernoulliNB()

#############GNB : 0.35
#clf = GaussianNB()

#############RF : 0.707
clf = RandomForestClassifier(n_estimators=200, criterion='entropy')
scores = cross_val_score(clf, X_train, y_train)
print scores.mean()


clf = svm.SVC(kernel='rbf', C=10, gamma=0.001, cache_size=500)
scores = testCV(clf, X_train, y_train, 5)

logit = LogisticRegression(penalty="l2", dual=True, C=1)
logit_scores = testCV(logit, X_train, y_train, 5)


pca = PCA(n_components=10)
pca.fit(X_train)
train_X_pca = pca.transform(X_train)
logit_scores = testCV(logit, train_X_pca, y_train, 5)

pca.components_
pca.explained_variance_ratio_

def testCV(clf, train_X, train_y, cvn):
	start_time = time.time()
	scores = cross_validation.cross_val_score(clf, train_X, train_y, cv=cvn)
	mean = "{:.5f}".format(scores.mean())
	sd = "{:.5f}".format(scores.std()*2)
	line = "accuracy: "+ str(mean) +" (+/- " + str(sd) + ")" +", "
	print line
	print("cross validation: --- %s seconds ---" % (time.time() - start_time))
	return scores