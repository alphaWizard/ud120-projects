#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### for initial visualization


# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
# plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
# plt.legend()
# plt.xlabel("bumpiness")
# plt.ylabel("grade")
# plt.show()

################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary



from time import time


## 1. naive bayes --- > 88.4%

# from sklearn.naive_bayes import GaussianNB as gnb
# from sklearn.metrics import accuracy_score
# clf = gnb()

# t0 = time()
# clf.fit(features_train,labels_train)
# print "training time:", round(time()-t0, 3), "s"

# t1 = time()
# prediction = clf.predict(features_test)
# print "prediction time:", round(time()-t1, 3), "s"

# accuracy = accuracy_score(labels_test,prediction)
# print "accuracy: ", (accuracy)



## 2. svm ----> 94.4%

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(C=100000, kernel="rbf")

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1=time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t1, 3), "s"

accuracy = accuracy_score(labels_test,pred)
print "accuracy: ", (accuracy)


## 3. decision tree ---> 92.4 

# from sklearn import tree
# from sklearn.metrics import accuracy_score

# clf = tree.DecisionTreeClassifier(min_samples_split=20)

# t0 = time()
# clf.fit(features_train, labels_train)
# print "training time:", round(time()-t0, 3), "s"

# t1=time()
# pred = clf.predict(features_test)
# print "prediction time:", round(time()-t1, 3), "s"

# accuracy = accuracy_score(labels_test,pred)
# print "accuracy: ", (accuracy)






try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
	pass
    
