""" Writing my first randomforest code.
Author : AstroDave
Date : 23rd September, 2012
please see packages.python.org/milk/randomforests.html for more

""" 

import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
import pylab as pl
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

csv_file_object = csv.reader(open('train.csv', 'rb')) #Load in the training csv file
header = csv_file_object.next() #Skip the fist line as it is a header
train_data=[] #Creat a variable called 'train_data'
for row in csv_file_object: #Skip through each row in the csv file
    train_data.append(row[1:]) #adding each row to the data variable
train_data = np.array(train_data) #Then convert from a list to an array

#I need to convert all strings to integer classifiers:
#Male = 1, female = 0:
train_data[train_data[0::,3]=='male',3] = 1
train_data[train_data[0::,3]=='female',3] = 0
#embark c=0, s=1, q=2
train_data[train_data[0::,10] =='C',10] = 0
train_data[train_data[0::,10] =='S',10] = 1
train_data[train_data[0::,10] =='Q',10] = 2

#I need to fill in the gaps of the data and make it complete.
#So where there is no price, I will assume price on median of that class
#Where there is no age I will give median of all ages

#All the ages with no data make the median of the data
train_data[train_data[0::,4] == '',4] = np.median(train_data[train_data[0::,4]\
                                           != '',4].astype(np.float))
#All missing ebmbarks just make them embark from most common place
train_data[train_data[0::,10] == '',10] = np.round(np.mean(train_data[train_data[0::,10] != '',10].astype(np.float)))


##### Code to add titles...
#print train_data[ ", Mr. " in train_data[:,2],  ]
#mrsLines =  train_data[", Mr. " in train_data[0::,2], 2]
#print mrsLines

mrIndex =  np.char.count( train_data[0::,2], ", Mr." ) > 0
msIndex =  np.char.count( train_data[0::,2], ", Ms." ) > 0
mrsIndex = np.char.count( train_data[0::,2], ", Mrs." ) > 0
otherIndex = np.logical_not( mrIndex + msIndex + mrsIndex )

train_data[ mrIndex, 2] = 1
train_data[ msIndex, 2] = 2
train_data[ mrsIndex, 2] = 3
train_data[ otherIndex, 2] = 4


#I need to do the same with the test data now so that the columns are in the same
#as the training data

test_file_object = csv.reader(open('test.csv', 'rb')) #Load in the test csv file
header = test_file_object.next() #Skip the fist line as it is a header
test_data=[] #Creat a variable called 'test_data'
ids = []
for row in test_file_object: #Skip through each row in the csv file
    ids.append(row[0])
    test_data.append(row[1:]) #adding each row to the data variable
test_data = np.array(test_data) #Then convert from a list to an array

#I need to convert all strings to integer classifiers:
#Male = 1, female = 0:
test_data[test_data[0::,2]=='male',2] = 1
test_data[test_data[0::,2]=='female',2] = 0
#ebark c=0, s=1, q=2
test_data[test_data[0::,9] =='C',9] = 0 #Note this is not ideal, in more complex 3 is not 3 tmes better than 1 than 2 is 2 times better than 1
test_data[test_data[0::,9] =='S',9] = 1
test_data[test_data[0::,9] =='Q',9] = 2

#All the ages with no data make the median of the data
test_data[test_data[0::,3] == '',3] = np.median(test_data[test_data[0::,3] != '',3].astype(np.float))
#All missing ebmbarks just make them embark from most common place
test_data[test_data[0::,9] == '',9] = np.round(np.mean(test_data[test_data[0::,9]\
                                                   != '',9].astype(np.float)))


#All the missing prices assume median of their respectice class
for i in xrange(np.size(test_data[0::,0])):
    if test_data[i,7] == '':
        test_data[i,7] = np.median(test_data[(test_data[0::,7] != '') &\
                                             (test_data[0::,0] == test_data[i,0])\
            ,7].astype(np.float))

# Titles mapping
mrIndex =  np.char.count( test_data[0::,1], ", Mr." ) > 0
msIndex =  np.char.count( test_data[0::,1], ", Ms." ) > 0
mrsIndex = np.char.count( test_data[0::,1], ", Mrs." ) > 0
otherIndex = np.logical_not( mrIndex + msIndex + mrsIndex )

test_data[mrIndex, 1] = 1
test_data[msIndex, 1] = 2
test_data[mrsIndex, 1] = 3
test_data[otherIndex, 1] = 4


train_data[:, 5] = (train_data[:,5].astype(np.float) + train_data[:,6].astype(np.float))
test_data[:, 4] = (test_data[:,4].astype(np.float) + test_data[:,5].astype(np.float))

## Removing some features
#remove embark (10 - train, 9 - test)
#remove age (4 - train, 3 - test)
# summing 5 and 6 -> deleting 6 (5 in test)
train_data = np.delete(train_data,[4,6,7,9,10],1) #remove the name data, cabin and ticket
test_data = np.delete(test_data,[3,5,6,8,9],1) #remove the name data, cabin and ticket

#The data is now ready to go. So lets train then test!
X = np.float32(train_data[:, 1:])
y = train_data[:, 0]
test_data = np.float32(test_data)

print 'Training '
nb = GaussianNB().fit(X, y)
nb_probas = nb.predict_proba(X)
nb_test_probas = nb.predict_proba(test_data)
print "Mean NB acc = ", np.mean( cross_validation.cross_val_score(nb, X, y, scoring="accuracy", cv=30, n_jobs=2))

forest = RandomForestClassifier(n_estimators=100, n_jobs=2, random_state=0)
forest = forest.fit(X, y)
forest_probas = forest.predict_proba(X)
forest_test_probas = forest.predict_proba(test_data)

extraTree = ExtraTreesClassifier(n_estimators=120, n_jobs=2, random_state=0) 
extraTree = extraTree.fit(X, y)
extraTree_probas = extraTree.predict_proba(X)
extraTree_test_probas = extraTree.predict_proba(test_data)

svc = SVC(C=10, probability=True, kernel="rbf", random_state=0)
svc = svc.fit(X,y)
svc_probas = svc.predict_proba(X)
svc_test_probas = svc.predict_proba(test_data)

lrc = LogisticRegression(C=1, random_state=0)
lrc = lrc.fit(X, y)
lrc_probas = extraTree.predict_proba(X)
lrc_test_probas = extraTree.predict_proba(test_data)


knn = KNeighborsClassifier(n_neighbors=1)
knn = knn.fit(X,y)
knn_probas = knn.predict_proba(X)
knn_test_probas = knn.predict_proba(test_data)

#X_extra = np.hstack((X, extraTree_probas))
#X_extra = np.hstack((X_extra, nb_probas))
#X_extra = np.hstack((X_extra, svc_probas))

print "Initial Mean SVC acc = ", np.mean(cross_validation.cross_val_score(svc, X, y, scoring="accuracy", cv=5, n_jobs=1))
#print "Mean Forest acc = ", np.mean(cross_validation.cross_val_score(forest, X, y, scoring="accuracy", cv=5, n_jobs=1))
print "Initial Mean ExtraTree acc = ", np.mean(cross_validation.cross_val_score(extraTree, X, y, scoring="accuracy", cv=5, n_jobs=1))
#print "Mean ExtraTree acc + extra = ", np.mean(cross_validation.cross_val_score(extraTree, X_extra, y, scoring="accuracy", cv=5, n_jobs=1))
print "Initial Mean LRC = ", np.mean(cross_validation.cross_val_score(lrc, X, y, scoring="accuracy", cv=5, n_jobs=1))
print "Initial Mean KNN = ", np.mean(cross_validation.cross_val_score(knn, X, y, scoring="accuracy", cv=5, n_jobs=1))

#X_probas = np.hstack( (forest_probas, extraTree_probas) )
#X_probas = np.hstack( (X_probas, nb_probas) )
#X_probas = np.hstack( (X_probas, svc_probas) )
#print "Mean Only Probas = ", np.mean( cross_validation.cross_val_score(extraTree, X_probas, y, scoring="accuracy", cv=5, n_jobs=1))

#X_test_probas = np.hstack( (forest_test_probas, extraTree_test_probas) )
#X_test_probas = np.hstack( (X_test_probas, nb_test_probas) )
#X_test_probas = np.hstack( (X_test_probas, svc_test_probas) )

def aumenta(X, y, test_data):

    #extraTree
    gridExtra = GridSearchCV(ExtraTreesClassifier(), {'n_estimators':[ 10, 100, 500], 'max_features':["auto", "log2", 1, 4] }).fit(X,y)
    extraTree = ExtraTreesClassifier(n_estimators=gridExtra.best_params_["n_estimators"], max_features=gridExtra.best_params_["max_features"], n_jobs=2, random_state=0).fit(X, y)
    extraTree_probas = extraTree.predict_proba(test_data)

    gridSVC = GridSearchCV(SVC(C=1), {'C':[ 1, 10, 100,1000,10000], 'gamma':[0,1,100,1000,10000]} ).fit(preprocessing.scale(X),y)
    svc_probas = SVC(probability=True, C=gridSVC.best_params_['C'], gamma=gridSVC.best_params_['gamma'] ).fit(preprocessing.scale(X),y).predict_proba(preprocessing.scale(test_data))

    gridLrc = GridSearchCV(lrc, {'C':[0.001, 1, 5, 10, 100, 200, 500, 10000], 'fit_intercept':[False, True]} ).fit(X,y)
    lrc_probas = LogisticRegression(C=gridLrc.best_params_['C'], fit_intercept=gridLrc.best_params_['fit_intercept']).fit(X, y).predict_proba(test_data)
    
    gridKnn = GridSearchCV(KNeighborsClassifier(), {'n_neighbors':[1,2,3,4,5,6,7,8,9,10,100, 10000]} ).fit(X,y)
    knn_probas = KNeighborsClassifier(n_neighbors=gridKnn.best_params_["n_neighbors"]).fit(X,y).predict_proba(test_data)
    
    thresold = 0.95

    # Test examples that are 0
    zeros = test_data[(extraTree_probas[:,0] >= thresold) & (svc_probas[:,0] >= thresold) & (lrc_probas[:,0] >= thresold) & (knn_probas[:,0] >= thresold)]
    # Test examples that are 1
    ones = test_data[(extraTree_probas[:,1] >= thresold) & (svc_probas[:,1] >= thresold) & (lrc_probas[:,1] >= thresold) & (knn_probas[:,1] >= thresold)]

    rest = test_data[(extraTree_probas[:,1] < thresold) & (svc_probas[:,1] < thresold) & (lrc_probas[:,1] < thresold) & (knn_probas[:,1] < thresold) & (extraTree_probas[:,0] < thresold) & (svc_probas[:,0] < thresold) & (lrc_probas[:,0] < thresold) & (knn_probas[:,0] < thresold)]

    before = X.shape[0]
    print "Before =", before
    X = np.vstack((X,zeros))
    X = np.vstack((X,ones))
    y = np.hstack((y, np.zeros(zeros.shape[0])))
    y = np.hstack((y, np.ones(ones.shape[0])))
    y = np.float32(y)
    
    print "Mean Default SVC = ", np.mean(cross_validation.cross_val_score(SVC(), X, y, scoring="accuracy", cv=5, n_jobs=1))
    print "Added =", X.shape[0] - before
    return X, y, rest

X, y, rest_test = aumenta(X, y, test_data)
i = 0
isIncreasing = True
lastSize = X.shape[0]

while isIncreasing:
    i += 1
    print "i = ", i
    X, y, rest_test = aumenta(X, y, rest_test)
    if lastSize == X.shape[0]:
        isIncreasing = False
    lastSize = X.shape[0]

def ensemble(lrc_probas, extra_probas, svm_probas, nb_probas, knn_probas):
    probas = lrc_probas + extra_probas + svm_probas + nb_probas + knn_probas
    probasFinal = probas[:,0] < probas[:,1]
    probasFinal = probasFinal.astype(int)
    return probasFinal

extra_probas = ExtraTreesClassifier(n_estimators=100, max_features=4, n_jobs=2, random_state=0).fit(X, y).predict_proba(test_data)
lrc_probas = LogisticRegression(C=100).fit(X, y).predict_proba(test_data)
nb_probas = GaussianNB().fit(X, y).predict_proba(test_data)
knn_probas = KNeighborsClassifier(n_neighbors=1).fit(X,y).predict_proba(test_data)

#X = preprocessing.scale(X)  # no gain
#X_norm = preprocessing.normalize(X)  # no gain in probability and loss in vanilla classifiers
#test_data_norm = preprocessing.normalize(test_data)

svm_probas = SVC(probability=True, C=1000, gamma=1).fit(preprocessing.scale(X), y).predict_proba(preprocessing.scale(test_data))

print 'Predicting'
output = ensemble(lrc_probas, extra_probas, svm_probas, nb_probas, knn_probas)

open_file_object = csv.writer(open("forest.csv", "wb"))
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))

