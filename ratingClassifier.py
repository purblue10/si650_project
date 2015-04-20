from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
from sklearn import svm
import numpy as np
import data_process_forrating as dp
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

train_path = "./data/train.json"
test_path = "./data/test.json"

train = dp.form_matrix(train_path)
test = dp.form_matrix(test_path)

train_X = [ row[1]  for row in train]
train_y = [ row[0]  for row in train]
test_X = [ row[1]  for row in train]
test_y = [ row[0]  for row in test]

train_X = np.array(train_X)
train_y = np.array(train_y)

#1-SVM-RBF
#clf = svm.SVC(kernel='rbf', C=1000, gamma=0.001, cache_size=1000)
#clf = svm.SVC(kernel='rbf', cache_size=1000)

#2-GNB
#clf = GaussianNB()

#3-MultinomialNB
#clf = MultinomialNB()

#4-BernoulliNB
#clf = BernoulliNB()

#5-LinearSVM
#clf = svm.SVC(kernel='linear')

#6-LogisticRegression
logit = LogisticRegression(penalty="l2", dual=True, C=1)

#Proper choice of C and gamma is critical to the SVM's performance.

'''model = svm.SVC(kernel='rbf', cache_size=1000)
param_grid = {
    "C": [0.0001,0.001, 0.01, 0.1,1, 10, 100,1000,10000],
    "gamma": [0.0001,0.001, 0.01, 0.1,1, 10, 100,1000,10000]
}
clf_grid = GridSearchCV(model, param_grid=param_grid, score_func=f1_score, cv=5)
clf_grid.fit(train_X, train_y)'''

#clf.fit(train_X, train_y)

scores = cross_validation.cross_val_score(logit, train_X, train_y,cv=5, scoring='f1_weighted')
scores.mean()
#clf_grid.best_estimator_
#clf_grid.grid_scores_
#clf_grid.best_score_

