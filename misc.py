import csv
import re 
import time
import data_process as dp
import numpy as np
import fs
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
from sklearn.metrics import f1_score
from sklearn.feature_extraction import DictVectorizer
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import roc_curve, auc

class LemmaTokenizer(object):
   def __init__(self):
      self.wnl = WordNetLemmatizer()
   def __call__(self, doc):
      return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def check_special_characters(sent):
   while(re.search('[^a-zA-Z0-9\-\+\'\"\#\*\@\!\?\/\&\(\)\:\;\,\.\s]',sent)):
      sent = re.sub('[^a-zA-Z0-9\-\+\'\"\#\*\@\!\?\/\&\(\)\:\;\,\.\s]','',sent)
   if re.search('\( *',sent):
      sent = re.sub('\( *', ' (',sent)
   if re.search(' *\)',sent):
      sent = re.sub(' *\)', ') ',sent)
   if re.search(',',sent):
      sent = re.sub('\,',' ', sent)
   return sent

def condensing(sent):
   # pattern = re.compile('[a-zA-Z0-9][\-\+\'\"\#\*\@\!\?\&\:\;\.][a-zA-Z0-9]')
   # while True:
   #    result = pattern.search(sent)
   #    if result==None:
   #       break
   #    index = result.start()+1
   #    sent=sent[:index]+sent[index+1:]
   # return re.sub('[^a-zA-Z0-9\s]',' ',sent).strip()
   sent = re.sub('[^a-zA-Z0-9\s]',' ',sent).strip()
   return re.sub('[0-9]','',sent).strip()

def setenceProcessing(sent):
   sent = check_special_characters(sent)
   sent = condensing(sent)
   return sent

def readData(path):
   data = []
   with open(path,'r') as reader:
      for line in reader:
         text = line[2:]
         text = check_special_characters(text)
         text = condensing(text)
         tp = (int(line[0]), text)
         data.append(tp)
   X = [row[1] for row in data]
   y = [row[0] for row in data]
   return X,y

def writeResult(path, result):
   output = [ (i+1, result[i]) for i in range(len(result))]
   writer = csv.writer(open(path, 'wb', buffering=0))
   writer.writerows([("Id","Category")])
   writer.writerows(output)


def getTextAndLabel(data):
   data_label = [ row['label'] for row in data]
   data_text = []
   for row in data:
      text = ""
      for line in data[0]['text']:
         text += line[0]+" "
      data_text.append(text)
   data_text = [" ".join([ line[0] for line in row['text']]) for row in data]
   return data_text, data_label

def getParameters(clf, feature_model):
   # feature model
   if type(feature_model) is PCA or type(feature_model) is RandomizedPCA:
      feature = 'pca'
      feature_param = feature_model.get_params()['n_components']
   elif  type(feature_model) is SelectKBest:
      feature = 'ch2'
      feature_param = feature_model.get_params()['k']
   # classifier
   if type(clf) is OneVsRestClassifier:
      try:
         c = clf.get_params()['estimator__estimator'].C
         gamma = clf.get_params()['estimator__estimator'].gamma
      except(KeyError):
         c = clf.get_params()['estimator__C']
         gamma = clf.get_params()['estimator__gamma']
   elif type(clf) is GridSearchCV:
      c = clf.best_params_['estimator__C']
      gamma = clf.best_params_['estimator__gamma']
   return [feature, str(feature_param), str(c), str(gamma)]

# For prediction, i.e. generate submission files
# path: FSMethod_FSparam_C_gamma
def prediction(clf, test_X, feature_model):
   start_time = time.time()
   params = getParameters(clf, feature_model)
   path = "./result/submission/"+"_".join(params)+".txt"
   prediction = clf.predict(test_X)
   result = prediction.tolist()
   writeResult(path, result)
   print("prediction: --- %s seconds ---" % (time.time() - start_time))
   print("path: "+path)
   

# for cross validation log
# use empty classifier
def runCV(clf, train_X, train_y, feature_model, cvn):
   start_time = time.time()
   path = "./result/eval_log.txt"
   params = getParameters(clf, feature_model)
   if type(clf) is GridSearchCV:
      clf = OneVsRestClassifier(svm.SVC(kernel='rbf', C=float(params[2]), gamma=float(params[3])))
   scores = cross_validation.cross_val_score(clf, train_X, train_y, cv=cvn)
   mean = "{:.5f}".format(scores.mean())
   sd = "{:.5f}".format(scores.std()*2)
   line = ",".join(params)+" - accuracy: "+ str(mean) +" (+/- " + str(sd) + ")" +", "
   line += str(scores)+" \n"
   writer = open(path, 'a')
   writer.write(line)
   writer.close()
   print("cross validation: --- %s seconds ---" % (time.time() - start_time))
   print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
   print("params: "+str(params))
   return scores

def testCV(clf, train_X, train_y, cvn):
   start_time = time.time()
   scores = cross_validation.cross_val_score(clf, train_X, train_y, cv=cvn)
   mean = "{:.5f}".format(scores.mean())
   sd = "{:.5f}".format(scores.std()*2)
   line = "accuracy: "+ str(mean) +" (+/- " + str(sd) + ")" +", "
   print line
   print("cross validation: --- %s seconds ---" % (time.time() - start_time))
   return scores

# mics.evaluation(prediction,test_y)
def evaluation(prediction, test_label, echo=True):
   n = len(test_label)
   answer = []
   for i in range(n):
   	if prediction[i]==test_label[i]:
   		answer.append((test_label[i], prediction[i],True))
   	else:
   		answer.append((test_label[i], prediction[i],False))
   numTrue = len([ row for row in answer if row[2]==True])
   numFalse = len([ row for row in answer if row[2]==False])
   numTrue_zero = len([ row for row in answer if row[2]==True and row[1]==0])
   numTrue_one = len([ row for row in answer if row[2]==True and row[1]==1])
   n_zero = len([ row for row in answer if row[1]==0])
   n_one = len([ row for row in answer if row[1]==1])
   if echo == True:
      print("Total Accuracy: %0.5f " % (numTrue/float(n)))
      print("0-label Accuracy: %0.5f (%d / %d)" % (numTrue_zero/float(n_zero), numTrue_zero, n_zero))
      print("1-label Accuracy: %0.5f (%d / %d)" % (numTrue_one/float(n_one), numTrue_one, n_one))

   return answer

def evaluation2(prediction, test_label, echo=True, process=False):
   n = len(test_label)
   prediction_label = prediction[0]
   prediction_prob = prediction[1]
   if process==True:
      prediction_prob = np.array([row.tolist()[0] if len(row.tolist())==1 else row.tolist() for row in prediction[1]])
   answer = []
   for i in range(n):
      if prediction_label[i]==test_label[i]:
         answer.append((test_label[i], prediction_label[i],True))
      else:
         answer.append((test_label[i], prediction_label[i],False))
   numTrue = len([ row for row in answer if row[2]==True])
   numFalse = len([ row for row in answer if row[2]==False])
   numTrue_zero = len([ row for row in answer if row[2]==True and row[1]==0])
   numTrue_one = len([ row for row in answer if row[2]==True and row[1]==1])
   n_zero = len([ row for row in answer if row[1]==0])
   n_one = len([ row for row in answer if row[1]==1])
   fpr, tpr, thresholds = roc_curve(test_label, prediction_prob[:, 1])
   roc_auc = auc(fpr, tpr)
   if echo == True:
      print("Total Accuracy: %0.5f " % (numTrue/float(n)))
      print("0-label Accuracy: %0.5f (%d / %d)" % (numTrue_zero/float(n_zero), numTrue_zero, n_zero))
      print("1-label Accuracy: %0.5f (%d / %d)" % (numTrue_one/float(n_one), numTrue_one, n_one))
      print "Area under the ROC curve : %f" % roc_auc
      print "Precision : %f,  Recall %f" % (numTrue_one/float(n_one), numTrue_one/float(numTrue_one+n_zero-numTrue_zero))
   return answer

def getData(train_path, test_path, typenum=0):
   start_time = time.time()
   train1, train2, train3 = dp.form_matrix(train_path, type=typenum)
   test1, test2, test3 = dp.form_matrix(test_path, type=typenum)
   
   ## dataset 1
   train_X1 = np.array([ row[1][0:1]+row[1][2:]  for row in train1])
   test_X1 = np.array([ row[1][0:1]+row[1][2:]  for row in test1])
   train_y = np.array([ row[0]  for row in train1])
   test_y = np.array([ row[0]  for row in test1])
   print("---Finish loading the first data")

   ## dataset 2
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
   test_X2 = dicVectorizer.transform(test_X2)
   print("---Finish loading the second data")
   ## dataset 3
   train_text, train_y3 = getTextAndLabel(train3)
   test_text, test_y3 = getTextAndLabel(test3)
   vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', lowercase=True, sublinear_tf=True, tokenizer=LemmaTokenizer(), ngram_range=(1,2))

   vectorizer.fit(train_text)
   train_X3 = vectorizer.transform(train_text)
   test_X3 = vectorizer.transform(test_text)
   print train_X3.shape
   ch2, train_X3, test_X3 = fs.chisq(train_X3, train_y3, test_X3, 20000)
   train_Xtot = np.hstack((train_X1, train_X2, train_X3.toarray()))
   test_Xtot = np.hstack((test_X1, test_X2, test_X3.toarray()))
   print("---Finish loading the third data")
   print("Finish loading data : --- %s seconds ---" % (time.time() - start_time))

   train_X = [train_X1, train_X2, train_X3, train_Xtot]
   test_X = [test_X1, test_X2, test_X3, test_Xtot]
   return train_X, test_X, train_y, test_y
