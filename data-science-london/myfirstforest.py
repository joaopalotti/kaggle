""" Writing my first randomforest code.
Author : AstroDave
Date : 23rd September, 2012
please see packages.python.org/milk/randomforests.html for more

""" 

import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier


csv_file_object = csv.reader(open('fulltrain.csv', 'rb')) #Load in the training csv file
train_data=[] #Creat a variable called 'train_data'
for row in csv_file_object: #Skip through each row in the csv file
    train_data.append(row[:]) #adding each row to the data variable
train_data = np.array(train_data) #Then convert from a list to an array

#I need to convert all strings to integer classifiers:
#Male = 1, female = 0:
#train_data[train_data[0::,3]=='male',3] = 1
#train_data[train_data[0::,3]=='female',3] = 0
#embark c=0, s=1, q=2
#train_data[train_data[0::,10] =='C',10] = 0
#train_data[train_data[0::,10] =='S',10] = 1
#train_data[train_data[0::,10] =='Q',10] = 2

#I need to fill in the gaps of the data and make it complete.
#So where there is no price, I will assume price on median of that class
#Where there is no age I will give median of all ages

#All the ages with no data make the median of the data
#train_data[train_data[0::,4] == '',4] = np.median(train_data[train_data[0::,4] != '',4].astype(np.float))
#All missing ebmbarks just make them embark from most common place
#train_data[train_data[0::,10] == '',10] = np.round(np.mean(train_data[train_data[0::,10] != '',10].astype(np.float)))


##### Code to add titles...
#print train_data[ ", Mr. " in train_data[:,2],  ]
#mrsLines =  train_data[", Mr. " in train_data[0::,2], 2]
#print mrsLines

#mrIndex =  np.char.count( train_data[0::,2], ", Mr." ) i> 0
#msIndex =  np.char.count( train_data[0::,2], ", Ms." ) > 0
#mrsIndex = np.char.count( train_data[0::,2], ", Mrs." ) > 0
#otherIndex = np.logical_not( mrIndex + msIndex + mrsIndex )

#train_data[ mrIndex, 2] = 1
#train_data[ msIndex, 2] = 2
#train_data[ mrsIndex, 2] = 3
#train_data[ otherIndex, 2] = 4


#I need to do the same with the test data now so that the columns are in the same
#as the training data

test_file_object = csv.reader(open('test.csv', 'rb')) #Load in the test csv file
test_data=[] #Creat a variable called 'test_data'
for row in test_file_object: #Skip through each row in the csv file
    test_data.append(row[:]) #adding each row to the data variable
test_data = np.array(test_data) #Then convert from a list to an array

#The data is now ready to go. So lets train then test!

print 'Training '
forest = RandomForestClassifier(n_estimators=100, n_jobs=2, random_state=0)
forest = forest.fit(train_data[0::,1::],\
                    train_data[0::,0])

extraTree = ExtraTreesClassifier(n_estimators=120, n_jobs=2, random_state=0) 
extraTree = extraTree.fit(train_data[0::,1::], train_data[0::,0])

from sklearn import cross_validation
print "Mean Forest acc = ", np.mean( cross_validation.cross_val_score(forest, train_data[:,1:], train_data[:,0], scoring="accuracy", cv=5, n_jobs=2))
print "Mean ExtraTree acc = ", np.mean( cross_validation.cross_val_score(extraTree, train_data[:,1:], train_data[:,0], scoring="accuracy", cv=5, n_jobs=2))


print 'Predicting'
output = forest.predict(test_data)
outputExtraTree = extraTree.predict(test_data)

open_file_object = csv.writer(open("forest.csv", "wb"))
#open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(output)

open_file_object = csv.writer(open("extraforest.csv", "wb"))
#open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(outputExtraTree)



#grid search for svm
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

svcmodel = SVC(c=1.0, kernel="rbf", degree=3, gamma=0.0, coef0=0.0, tol=0.001, max_iter=-1)
svcmodel.fit(train_data[0::,1:], train_data[0::,0])

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
gridmodel = GridSearchCV(SVC(C=1), tuned_parameters, scoring="accuracy", n_jobs=-1)
gridmodel.fit(X, y)

