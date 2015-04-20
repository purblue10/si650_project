#####Author: Wizard
#Note:
#Function return to 3 matrix: clf1, clf2, clf3
#@param:
#	k: default k = 0, all json file will be processed
#	type: return only selected type of data(clf1, clf2, or clf3). Default type=0 to return all data
#clf1 = [label, [quality, avg_grade, helpfulness, clarity, easiness, avg_interest, avg_textBookUse, avg_takenCredit, num_of_rating]]
#clf2 = [label, [tags]]
#clf3 = [label, str]

import json
import random
import numpy
import time
import misc

def form_matrix(inFile, k=0, type=0):
	if type>3 or type<0:
		print "type value error (should be 0, 1, 2, or 3)"
		return None
	start_time = time.time()
	count = 0
	clf1 = []
	clf2 = []
	clf3 = []
	with open(inFile) as data_file:
		data = data_file.readline()
		while data:
			#print data
			if k == 0 or count < k: 
				data_json = json.loads(data)
				tid = data_json['tid']
				toAdd1 = [0, []]
				toAdd2 = [0, []]
				toAdd3 = {'label':0, 'text':[], 'tid':tid, }
				if data_json['hotness'] != 'cold-chili':
					toAdd1[0] = toAdd2[0] =  1
					toAdd3['label'] = 1
				toAdd1[1].append(float(data_json['quality']))
				toAdd1[1].append(data_json['avg_grade'])
				toAdd1[1].append(float(data_json['helpfulness']))
				toAdd1[1].append(float(data_json['clarity']))
				toAdd1[1].append(float(data_json['easiness']))
				Interest = 0
				TextBookUse = 0
				TakenCredit = 0
				NumRate = 0
				for rating in data_json['ratings']:
					if rating['teacherRatingTags']:
						toAdd2[1] += rating['teacherRatingTags']

					if rating['rInterest'] == 'Meh':
						Interest += 1.0
					elif rating['rInterest'] == 'Low':
						Interest += 2.0
					elif rating['rInterest'] == 'Sorta interested':
						Interest += 3.0
					elif rating['rInterest'] == 'Really into it':
						Interest += 4.0
					elif rating['rInterest'] == 'It\'s my life':
						Interest += 5.0
					
					if rating['rTextBookUse'] == 'What textbook?':
						TextBookUse += 1.0
					elif rating['rTextBookUse'] == 'Barely cracked it open':
						TextBookUse += 2.0
					elif rating['rTextBookUse'] == 'You need it sometimes':
						TextBookUse += 3.0
					elif rating['rTextBookUse'] == 'It\'s a must have':
						TextBookUse += 4.0
					elif rating['rTextBookUse'] == 'Essential to passing':
						TextBookUse += 5.0

					if rating['takenForCredit'] != 'Yes':
						TakenCredit += 1.0
					
					if type==0 or type==3:
						comment = misc.setenceProcessing(rating['rComments'])
						tp = (comment, rating['helpCount'], rating['notHelpCount'])
						toAdd3['text'].append(tp)

					NumRate += 1.0

				toAdd1[1].append(Interest/NumRate)
				toAdd1[1].append(TextBookUse/NumRate)
				toAdd1[1].append(TakenCredit/NumRate)
				toAdd1[1].append(NumRate)

				data = data_file.readline()
				count += 1
				clf1.append(toAdd1)
				clf2.append(toAdd2)
				clf3.append(toAdd3)
			else:
				break
	print("Numer of lines: %d lines" % (count))
	print("--- %s seconds ---" % (time.time() - start_time))
	if(type==0):
		return clf1, clf2, clf3
	elif(type==1):
		return clf1
	elif(type==2):
		return clf2
	elif(type==3):
		return clf3



# mtx1, mtx2, mtx3 = form_matrix('umich.json', 1)
# print mtx1
# print mtx2
# print mtx3


# total number of data: 33565
# cold-chili: 23905 (70%) / mean: 12.7, max: 737, min:1
# > cold : 9660 (30%) / mean: 10.5, max:426, min:1
# warm-chili: 8888
# steamy-chili: 589
# scorching-chili: 184

def getTrainTestPairs(path):
	reader = open(path, 'r')
	hotness_tid = {}
	data = {}
	for line in reader:
		js = json.loads(line)
		hot = js['hotness']
		tid = js['tid']
		N_ratings = len(js['ratings'])
		tp = (tid, N_ratings)
		if hot not in hotness_tid:
			hotness_tid[hot] = [tp]
		else:
			hotness_tid[hot].append(tp)
		data[tid] = js

	cold = [  tp for tp in hotness_tid['cold-chili'] ]
	warm = [  tp for tp in hotness_tid['warm-chili'] ]
	steamy = [  tp for tp in hotness_tid['steamy-chili'] ]
	scorching = [  tp for tp in hotness_tid['scorching-chili'] ]
	hot = warm + steamy + scorching
	total = cold + hot

	cold_test = random.sample(set(cold), 4900)
	hot_test = random.sample(set(hot), 2100)

	cold_stat=[numpy.mean([  tp[1] for tp in cold_test]), numpy.max([  tp[1] for tp in cold_test]), numpy.min([  tp[1] for tp in cold_test])]
	hot_stat=[numpy.mean([  tp[1] for tp in hot_test]), numpy.max([  tp[1] for tp in hot_test]), numpy.min([  tp[1] for tp in hot_test])]
	print "cold statistics in test: ", cold_stat
	print "hot statistics in test", hot_stat

	test_tp = cold_test+hot_test
	test_tids = set([ tp[0] for tp in test_tp])
	train = []
	test = []
	for key in data.keys():
		if key in test_tids:
			test.append(data[key])
		else:
			train.append(data[key])
	return train, test

def writerTrainTest(path):
	train, test = getTrainTestPairs(path)
	writer = open("./data/train.json", 'w')
	for line in train:
		writer.write(json.dumps(line)+"\n")
	writer.close()
	writer = open("./data/test.json", 'w')
	for line in test:
		writer.write(json.dumps(line)+"\n")
	writer.close()



#data_process.writerTrainTest("./data.json")
