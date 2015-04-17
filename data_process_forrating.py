#####Author: Wizard
###modified yzhgong
#default k = 0, all json file will be processed
#clf1 = [label, [quality, avg_grade, helpfulness, clarity, easiness, avg_interest, avg_textBookUse, avg_takenCredit, num_of_rating]]

import json,csv
#from sklearn import svm
#import numpy as np

def form_matrix(inFile, k=0):
	count = 0
	clf1 = []
	#clf2 = []
	#clf3 = []
	with open(inFile) as data_file:
		data = data_file.readline()
		while data:
			#print data
			if k == 0 or count < k: 
				data_json = json.loads(data)
				toAdd1 = [0, []]
				#toAdd2 = [0, []]
				#toAdd3 = [0, '']
				if data_json['hotness'] != 'cold-chili':
					toAdd1[0] = 1
					#toAdd2[0] = toAdd3[0] = 1
				toAdd1[1].append(float(data_json['quality']))
				avg_grade = data_json['avg_grade']
				grade = ['N/A','A+','A','A-','B+','B','B-','C+','C','C-','D+','D','D-','F']
				grade_p = [3.0,4.33,4.0,3.67,3.33,3.0,2.67,2.33,2.0,1.67,1.33,1.0,0.67,0.0]
				try:
					for i in range(14):
						if avg_grade == grade[i]:
							toAdd1[1].append(float(grade_p[i]))
				except:
					toAdd1[1].append(float(grade_p[0]))
				#toAdd1[1].append(data_json['avg_grade'])
				toAdd1[1].append(float(data_json['helpfulness']))
				toAdd1[1].append(float(data_json['clarity']))
				toAdd1[1].append(float(data_json['easiness']))
				Interest = 0
				TextBookUse = 0
				TakenCredit = 0
				NumRate = 0
				for rating in data_json['ratings']:
					#if rating['teacherRatingTags']:
					#	toAdd2[1] += rating['teacherRatingTags']

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
					
					#toAdd3[1] += rating['rComments'] + ' '					

					NumRate += 1.0

				toAdd1[1].append(float(Interest/NumRate))
				toAdd1[1].append(float(TextBookUse/NumRate))
				toAdd1[1].append(float(TakenCredit/NumRate))
				toAdd1[1].append(float(NumRate))

				data = data_file.readline()
				count += 1
				clf1.append(toAdd1)
				#clf2.append(toAdd2)
				#clf3.append(toAdd3)
			else:
				break

	return clf1
	#return clf1, clf2, clf3

#mtx1, mtx2, mtx3 = form_matrix('umich.json', 1)
mtx1 = form_matrix('train.json', 0)

# total number of data: 33565
# cold-chili: 23905 (70%) / mean: 12.7, max: 737, min:1
# > cold : 9660 (30%) / mean: 10.5, max:426, min:1
# warm-chili: 8888
# steamy-chili: 589
# scorching-chili: 184

output_csv = csv.writer(open('train.csv','wb',buffering=0))
output_test_cate = csv.writer(open('train_class.csv','wb',buffering=0))


for line in mtx1:
    output_csv.writerow(line[1])
    output_test_cate.writerow(line)

#X = np.array(X)

#clf = svm.SVC()
#clf.fit(X, Y)