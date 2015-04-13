#####Author: Wizard
#Note:
#Function return to 3 matrix: clf1, clf2, clf3
#default k = 0, all json file will be processed
#clf1 = [label, [quality, avg_grade, helpfulness, clarity, easiness, avg_interest, avg_textBookUse, avg_takenCredit, num_of_rating]]
#clf2 = [label, [tags]]
#clf3 = [label, str]

import json

def form_matrix(inFile, k=0):
	
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
				toAdd1 = [0, []]
				toAdd2 = [0, []]
				toAdd3 = [0, '']
				if data_json['hotness'] != 'cold-chili':
					toAdd1[0] = toAdd2[0] = toAdd3[0] = 1
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

					if rating['takenForCredit'] != 'N/A':
						TakenCredit += 1.0
					
					toAdd3[1] += rating['rComments'] + ' '					

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

	return clf1, clf2, clf3


mtx1, mtx2, mtx3 = form_matrix('umich.json', 1)
print mtx1
print mtx2
print mtx3
	