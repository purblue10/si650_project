# Hotness classifier for RateMyProfessor

## Feautres
There are three classifiers for each type of data, and here are specific lists of features for each classifier.

Label:
  hotness: Hotness (binary, either hot or not hot)

For classifier1:
  quality: Overall Quality [1-5]
  avg_grade: Average Grade [F-A+]
  helpfulness: Helpfulness [1-5]
  clarity: Clarity [1-5]
  easiness: Easiness [1-5]
  rInterest (in ratings): level of interest [1-5]
  rTextBookUse (in ratings): level of textbook use in class [1-5]
  takenForCredit (in ratings): binary (either yes or no)

For classifier2:
  teacherRatingTags (in ratings): list of tags
  
For classifier3:
  rComments (in ratings): list of comments

Additional information:
  helpCount / notHelpCount (in ratings)
  attendance (in ratings): binary
  onlineClass (in ratings): binary
  rDate (in ratings)

## Classifiers


