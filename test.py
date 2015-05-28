import csv
import re 
import misc
import time
import fs
import data_process as dp
from sklearn.feature_extraction import DictVectorizer
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

class LemmaTokenizer(object):
	def __init__(self):
		self.wnl = WordNetLemmatizer()
	def __call__(self, doc):
		return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]



# train: category, text
# text: id, text

train_path = "./data/data2/train.json"
test_path = "./data/data2/test.json"

train = dp.form_matrix(train_path, type=2)
test = dp.form_matrix(test_path, type=2)

train1 = [ [row[0]]+row[1] for row in train]

writer = csv.writer(open("train1.csv", "wb"))
writer.writerows(train1)

test1 = [ [row[0]]+row[1] for row in test]
writer = csv.writer(open("test1.csv", "wb"))
writer.writerows(test1)

train_text, train_y = misc.getTextAndLabel(train)
test_text, test_y = misc.getTextAndLabel(test)

for i, item in enumerate(train):
	if len(item[1])>6:
		break

train_X2 = []
test_X2 = []
for item in train:
   dic = {}
   for tag in item[1]:
      dic[tag] = 1 if  tag not in dic else dic[tag]+1
   train_X2.append(dic)

for item in test:
   dic = {}
   for tag in item[1]:
      dic[tag] = 1 if  tag not in dic else dic[tag]+1
   test_X2.append(dic)

dicVectorizer = DictVectorizer(sparse=False)
train2 = dicVectorizer.fit_transform(train_X2)
test2 = dicVectorizer.transform(test_X2)

train2 = train2.tolist()
test2 = test2.tolist()
head = dicVectorizer.get_feature_names()

writer = csv.writer(open("train2.csv", "wb"))
writer.writerows([head])
writer.writerows(train2)

writer = csv.writer(open("test2.csv", "wb"))
writer.writerows([head])
writer.writerows(test2)
