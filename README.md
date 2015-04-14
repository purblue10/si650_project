# Hotness classifier for RateMyProfessor

## Data
Total number of data: 33565
 * cold-chili: 23905 (70%) / # number of ratings: mean=12.7, max=737, min=1
 * hot-chili : 9660 (30%) / # number of ratings: mean=10.5, max=426, min=1
 ** warm-chili: 8888
 ** steamy-chili: 589
 ** scorching-chili: 184

**Train data: 30065 (cold:21455, hot:8610)**
**Test data: 3500 (cold:2450, hot:1050)**

## Feautres
There are three classifiers for each type of data, and here are specific lists of features for each classifier. <br> (feature name: descriptions)

**Label:**
  * hotness: Hotness [cold-chili, warm-chili, steamy-chili, scorching-chili]

**For classifier1:**
  * quality: Overall Quality [1-5]
  * avg_grade: Average Grade [F-A+]
  * helpfulness: Helpfulness [1-5]
  * clarity: Clarity [1-5]
  * easiness: Easiness [1-5]
  * rInterest (in ratings): level of interest [1-5]
  * rTextBookUse (in ratings): level of textbook use in class [1-5]
  * takenForCredit (in ratings): binary (either yes or no)

**For classifier2:**
  * teacherRatingTags (in ratings): list of tags
  
**For classifier3:**
  * rComments (in ratings): list of comments

**Additional information:**
  * helpCount / notHelpCount (in ratings)
  * attendance (in ratings): binary
  * onlineClass (in ratings): binary
  * rDate (in ratings)

## Classifiers


