""" Writing my first randomforest code.
Author : AstroDave
Date : 23rd September, 2012
please see packages.python.org/milk/randomforests.html for more

""" 

#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
#gridmodel = GridSearchCV(SVC(C=1), tuned_parameters, scoring="accuracy", n_jobs=-1)
#gridmodel.fit(X, y)

import sys
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing


csv_file_object = csv.reader(open('fulltrain.csv', 'rb')) #Load in the training csv file
train_data=[] #Creat a variable called 'train_data'
for row in csv_file_object: #Skip through each row in the csv file
    train_data.append(row[:]) #adding each row to the data variable
train_data = np.array(train_data) #Then convert from a list to an array

test_file_object = csv.reader(open('test.csv', 'rb')) #Load in the test csv file
test_data=[] #Creat a variable called 'test_data'
for row in test_file_object: #Skip through each row in the csv file
    test_data.append(row[:]) #adding each row to the data variable
test_data = np.array(test_data) #Then convert from a list to an array

#The data is now ready to go. So lets train then test!
print 'Training '
X = np.float64(train_data[0::,1:])
y = train_data[0::, 0]
test_data = np.float64(test_data)


def pcatransf(X, components=12):
    pca = PCA(n_components=components,whiten=True)
    pca.fit(X)
    return pca, pca.transform(X)

def ensemble(lrc_probas, extra_probas, svm_probas, nb_probas, knn_probas):
    probas = lrc_probas + extra_probas + svm_probas + nb_probas + knn_probas
    probasFinal = probas[:,0] < probas[:,1]
    probasFinal = probasFinal.astype(int)
    return probasFinal

def gridSearchTestData(X, y, test):

    print "Test.shape ---> ", test.shape
    gridExtra = GridSearchCV(ExtraTreesClassifier(), {'n_estimators':[ 10, 100, 500], 'max_features':["auto", "log2", 1, 4] }, n_jobs=-1).fit(X,y)
    extra_probas = ExtraTreesClassifier(n_estimators=gridExtra.best_params_["n_estimators"], max_features=gridExtra.best_params_["max_features"], n_jobs=10, random_state=0).fit(X, y).predict_proba(test)

    gridSVC = GridSearchCV(SVC(C=1), {'C':[ 1, 10, 100,1000,10000], 'gamma':[0,1,100,1000,10000]}, n_jobs=-1 ).fit(preprocessing.normalize(X),y)
    svc_probas = SVC(probability=True, C=gridSVC.best_params_['C'], gamma=gridSVC.best_params_['gamma'] ).fit(preprocessing.normalize(X),y).predict_proba(preprocessing.normalize(test))

    gridLrc = GridSearchCV(LogisticRegression(), {'C':[0.001, 1, 5, 10, 100, 200, 500, 10000], 'fit_intercept':[False, True]}, n_jobs=10 ).fit(X,y)
    lrc_probas = LogisticRegression(C=gridLrc.best_params_['C'], fit_intercept=gridLrc.best_params_['fit_intercept']).fit(X, y).predict_proba(test)

    gridKnn = GridSearchCV(KNeighborsClassifier(), {'n_neighbors':[1,2,3,4,5,6,7,8,9,10,100, 10000]}, n_jobs=10 ).fit(X,y)
    knn_probas = KNeighborsClassifier(n_neighbors=gridKnn.best_params_["n_neighbors"]).fit(X,y).predict_proba(test)

    nb_probas = GaussianNB().fit(X, y).predict_proba(test)
    
    #print "Shapes --> ", extra_probas.shape, svc_probas.shape, lrc_probas.shape, knn_probas.shape, nb_probas.shape

    return extra_probas, svc_probas, lrc_probas, knn_probas, nb_probas
    #return [], svc_probas, [], [], []
def aumenta(X, y, test):
    
    extra_probas, svc_probas, lrc_probas, knn_probas, nb_probas = gridSearchTestData(X,y,test)
    #print "Shapes --> ", extra_probas.shape, svc_probas.shape, lrc_probas.shape, knn_probas.shape, nb_probas.shape
    print "Shapes --> ", svc_probas.shape
    thresold = 0.99
    #extraThresold = 0.99

    # Test examples that are 0
    #zeros = test[(svc_probas[:,0] >= thresold)]# | (extra_probas[:,0] >= thresold)] #| (lrc_probas[:,0] >= thresold) | (knn_probas[:,0] >= thresold) | (nb_probas[:,0] >= thresold)]
    zeros = test[(svc_probas[:,0] >= thresold) | (extra_probas[:,0] >= thresold) | (lrc_probas[:,0] >= thresold) | (knn_probas[:,0] >= thresold) | (nb_probas[:,0] >= thresold)]
    # Test examples that are 1
    ones = test[(svc_probas[:,1] >= thresold) | (extra_probas[:,1] >= thresold) | (lrc_probas[:,1] >= thresold) | (knn_probas[:,1] >= thresold) | (nb_probas[:,1] >= thresold) ]

    ##TODO: check if the same guy is in ones and zeros
    
    rest = test[((svc_probas[:,1] < thresold) & (svc_probas[:,0] < thresold))] # & (svc_probas[:,1] < thresold) & (lrc_probas[:,1] < thresold) & (knn_probas[:,1] < thresold) & (nb_probas[:,1] < thresold)) & ((extra_probas[:,0] < thresold) & (svc_probas[:,0] < thresold) & (lrc_probas[:,0] < thresold) & (knn_probas[:,0] < thresold) & (nb_probas[:,0] < thresold) )]

    before = X.shape[0]
    print "Before =", before
    X = np.vstack((X,zeros))
    X = np.vstack((X,ones))
    y = np.hstack((y, np.zeros(zeros.shape[0])))
    y = np.hstack((y, np.ones(ones.shape[0])))
    y = np.float32(y)
    
    print "Mean Default SVC = ", np.mean(cross_validation.cross_val_score(SVC(), X, y, scoring="accuracy", cv=5, n_jobs=-1))
    print "Added =", X.shape[0] - before
    print "Rest.shape = ", rest.shape
    return X, y, rest


#svcmodel.fit(X,y)
#svcProbas = svcmodel.predict_proba(test_data)
#probas = extraProbas + forestProbas + svcProbas 
#probasFinal = probas[:,0:1] < probas[:,1:2]
#probasFinal = probasFinal.astype(int)
#extraFeaturesTrain = np.hstack((X, extraProbasTrain))
#extraFeaturesTest = np.hstack((test_data, extraProbasTest))

#X_pca = npcatransf(X)
#X_pca_extra = pcatransf(extraFeaturesTrain)

#X_pca_test = pcatransf(test_data)
#X_pca_extra_test = pcatransf(extraFeaturesTest)

#print "Mean ExtraTree acc = ", np.mean( cross_validation.cross_val_score(SVC(C=1), X, y, scoring="accuracy", cv=5, n_jobs=2))
#print "Mean ExtraTree acc PCA = ", np.mean( cross_validation.cross_val_score(SVC(C=1), X_pca, y, scoring="accuracy", cv=5, n_jobs=2))
#print "Mean ExtraTree acc PCA + extra = ", np.mean( cross_validation.cross_val_score(SVC(C=1), X_pca_extra, y, scoring="accuracy", cv=5, n_jobs=2))

#svcmodel = SVC(C=1.0, kernel="rbf", degree=3, gamma=0.0, coef0=0.0, tol=0.001, max_iter=-1, probability=True)
#svcpredicts = svcmodel.fit(X_pca_extra, y).predict_proba(X_pca_extra)

#svcextraFeaturesTrain = np.hstack((X_pca_extra, svcpredicts))
#print "Mean ExtraTree acc PCA + extra + svc= ", np.mean( cross_validation.cross_val_score(SVC(C=1), svcextraFeaturesTrain, y, scoring="accuracy", cv=5, n_jobs=2))

#svcmodel = SVC(C=1.0, kernel="rbf", degree=3, gamma=0.0, coef0=0.0, tol=0.001, max_iter=-1, probability=True)
#svcpredicts = svcmodel.fit(X_pca_extra, y).predict_proba(X_pca_extra_test)

#X_pca_extra_svc_test = np.hstack((X_pca_extra_test, svcpredicts))


#PCA transformation
pca, X = pcatransf(X)
test_data = pca.transform(test_data)

print "Initial Mean NB acc = ", np.mean( cross_validation.cross_val_score(GaussianNB(), X, y, scoring="accuracy", cv=5, n_jobs=-1))
print "Initial Mean SVC acc = ", np.mean(cross_validation.cross_val_score(SVC(), X, y, scoring="accuracy", cv=5, n_jobs=-1))
print "Initial Mean ExtraTree acc = ", np.mean(cross_validation.cross_val_score(ExtraTreesClassifier(), X, y, scoring="accuracy", cv=5))
print "Initial Mean LRC = ", np.mean(cross_validation.cross_val_score(LogisticRegression(), X, y, scoring="accuracy", cv=5, n_jobs=-1))
print "Initial Mean KNN = ", np.mean(cross_validation.cross_val_score(KNeighborsClassifier(), X, y, scoring="accuracy", cv=5, n_jobs=-1))

X, y, rest_test = aumenta(X, y, test_data)
i = 0
isIncreasing = True
lastSize = X.shape[0]

while isIncreasing and i < 10:
    i += 1
    print "i = ", i
    X, y, rest_test = aumenta(X, y, rest_test)
    if lastSize == X.shape[0]:
        isIncreasing = False
    lastSize = X.shape[0]


#extra_probas, svc_probas, lrc_probas, knn_probas, nb_probas = gridSearchTestData(X, y, test_data)
#finalPrediction = ensemble(lrc_probas, extra_probas, svc_probas, nb_probas, knn_probas)

svcmodel = GridSearchCV(SVC(C=1), {'C':[ 1, 10, 100,1000,10000], 'gamma':[0,1,100,1000,10000]}, n_jobs=-1 ).fit(preprocessing.normalize(X),y)
#svcmodel = SVC(C=1.0, kernel="rbf", degree=3, gamma=1, coef0=0.0, tol=0.001, max_iter=-1, probability=False).fit(preprocessing.normalize(X),y)
#print "Initial Mean SVC acc = ", np.mean(cross_validation.cross_val_score(svcmodel, preprocessing.normalize(X) , y, scoring="accuracy", cv=5))
#predictstrain = svcmodel.predict(preprocessing.normalize(X))
predictstest = svcmodel.predict(preprocessing.normalize(test_data))

newX = np.vstack((X,test_data))
newY = np.ravel(np.vstack((y.reshape(y.shape[0],1),predictstest.reshape(9000,1))))

svcmodel = SVC(C=10, gamma=0).fit(newX,newY)
print "using also predictions SVC acc = ", np.mean(cross_validation.cross_val_score(svcmodel, newX , newY, scoring="accuracy", cv=5))
#svcmodel = SVC(C=1.0, kernel="rbf", degree=3, gamma=0.0, coef0=0.0, tol=0.001, max_iter=-1, probability=True).fit(newX,y)
svcmodel = GridSearchCV(SVC(C=1), {'C':[ 1, 10, 100,1000,10000], 'gamma':[0,1,100,1000,10000]}, n_jobs=-1 ).fit(newX,newY)
finalPrediction = svcmodel.predict(test_data)


# the same thing for the test data
#extraTree = ExtraTreesClassifier(n_estimators=120, n_jobs=2, random_state=0).fit(X, y)
#extraProbasTest = extraTree.predict_proba(test_data)
#extraFeaturesTest = np.hstack((test_data, extraProbasTest))
#X_pca_extra_test = pcatransf(extraFeaturesTest)
#svcpredicts = svcmodel.fit(X_pca_extra, y).predict_proba(X_pca_extra)
#svcextraFeaturesTrain = np.hstack((X_pca_extra, svcpredicts))
#svcpredicts = svcmodel.fit(X_pca_extra_test, svcPredicts).predict_proba(X_pca_extra_test)


#finalPrediction = svcmodel.fit(svcextraFeaturesTrain, y).predict(X_pca_extra_svc_test).reshape(9000,1)
print 'Predicting'
#output = forest.predict(test_data)

ids = np.arange(1,9001)
output = zip(ids, finalPrediction.astype(int))


outfile = open("outfile.csv", "wb")
open_file_object = csv.writer(outfile)
open_file_object.writerow(["Id","Solution"])
open_file_object.writerows(output)
outfile.close()


