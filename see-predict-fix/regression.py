import csv as csv
import math, sys
from collections import Counter
import numpy as np
from datetime import datetime

from sklearn.cross_validation import cross_val_score, KFold, train_test_split

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, CountVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression, RidgeCV, Ridge, Lasso, MultiTaskLasso, LassoLars, OrthogonalMatchingPursuit, ARDRegression, SGDRegressor, PassiveAggressiveRegressor
from sklearn.metrics import make_scorer
from sklearn.isotonic import IsotonicRegression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import OneHotEncoder, scale, normalize
from sklearn.grid_search import GridSearchCV

normalizeit = True
trainingStats = True

#inTrain = csv.reader( open("smalltrain.csv", "rb") )
inTrain = csv.reader( open("train.csv", "rb") )
header = inTrain.next()
data = [row for row in inTrain]

goal = []
views = []
votes = []
comments = []
times = []
descriptions = []
summaries = []
locations = []
tags = []
alltext = []
lastTime = datetime.strptime("2013-04-30 23:51:37", "%Y-%m-%d %H:%M:%S")

#Latitude and longitudes:
#Chicago ==> 41.8819, -87.6278
#New Haven=> 41.3100, -72.9236
#Ockland ==> 37.8044, -122.2708
#Richmond==> 37.5410, -77.4329
city = []
minDists = []
sourceL = []

def geoDist(lat, log, nlat, nlog):
    return math.sqrt( (lat-nlat) * (lat-nlat) + (log-nlog) * (log-nlog)  )

def findClosestCity(lat, log):
    chi = geoDist(lat,log, 41.8819, -87.6278) #Chicago
    new = geoDist(lat,log, 41.3100, -72.9236)
    ock = geoDist(lat,log, 37.8044, -122.2708)
    ric = geoDist(lat,log, 37.5410, -77.4329)
    minDist = np.min([chi,new,ock,ric])
    if minDist == chi:
        return [0], minDist
    elif minDist == new:
        return [1], minDist
    elif minDist == ock:
        return [2], minDist
    elif minDist == ric:
        return [3], minDist

def findSource(source):
    if source  == 'NA': 
        return [0]
    elif source == 'remote_api_created':
        return [1]
    elif source == 'city_initiated':
        return [3]
    elif source == 'web':
        return [4]
    elif source == 'New Map Widget':
        return [5]
    elif source == 'Map Widget':
        return [5]
    elif source == 'android':
        return [6]
    elif source == 'Mobile Site': 
        return [6]
    elif source == 'iphone':
        return [6] 

for row in data[0:-1:1]:
#for row in data:
    #"id","latitude","longitude","summary","description","num_votes","num_comments","num_views","source","created_time","tag_type"
    id, latitude, longitude, summary, description, num_votes, num_comments, num_views, source, created_time, tag_type = row

    time = datetime.strptime(created_time, "%Y-%m-%d %H:%M:%S")
    timediff = (lastTime - time).days

    votes.append(num_votes)
    comments.append(num_comments)
    views.append(num_views)
    times.append(timediff)
    tags.append(tag_type)
    
    alltext.append(summary + " " + description + " " + tag_type)
    summaries.append(summary)
    descriptions.append(description)
    
    # lattitude and longitude measures
    cityName, minDist = findClosestCity(float(latitude), float(longitude))
    city.append(cityName)
    minDists.append(minDist)
    sourceL.append(findSource(source))


ohe = OneHotEncoder()
sohe = OneHotEncoder()
Xcity = ohe.fit_transform(city).toarray()
Xsource = sohe.fit_transform(sourceL).toarray()
print "Vectorizing"
XminDists = np.array(minDists)

#vectorizer = HashingVectorizer(n_features=100, strip_accents="ascii")  # training: 0.971392246367
#vectorizer = HashingVectorizer(n_features=1000, strip_accents="ascii")
#vectorizer = HashingVectorizer(n_features=2000, strip_accents="ascii")
vectorizer = CountVectorizer(max_features=100, strip_accents="ascii")#, analyzer='char_wb') #seems good for big data
tfidf = TfidfTransformer()

#dt = vectorizer.fit_transform(descriptions).toarray()
#st = vectorizer.fit_transform(summaries).toarray()
#tt = vectorizer.fit_transform(tags).toarray()
#text = vectorizer.fit_transform(alltext).toarray()
text = tfidf.fit_transform(vectorizer.fit_transform(alltext).toarray()).toarray()
times = scale(np.array(times).astype(int))

#X = np.column_stack((text,times,Xcity))
sourcepca = PCA(n_components=5, whiten=False)
textpca = PCA(n_components=20)
#text = textpca.fit_transform(text)
#svd = TruncatedSVD(n_components=20, random_state=29)
#text = svd.fit_transform(text)

#X = np.column_stack((text,times))
#X = np.column_stack((text,times,Xcity,XminDists))

Xsource = sourcepca.fit_transform(Xsource)
#X = np.column_stack((text,times,Xcity,Xsource)) # XminDists is bad for small data
X = np.column_stack((text,times,Xcity,XminDists,Xsource))
#X = np.column_stack((text,times,Xcity,XminDists,Xsource))
#X = np.column_stack((text,times,Xcity,XminDists))
#X = np.column_stack((dt,st,tt,times))
y_votes = np.array(votes).astype(int)
y_comments = np.array(comments).astype(int)
y_views = np.array(views).astype(int)

if normalizeit:
    X = normalize(X)
#X = svd.fit_transform(X)
#X = pca.fit_transform(X)

print "X created"

def calculateRMSLE(actuals, predictions):
    predictions[ predictions < 0 ] = 0.0
    return math.sqrt( 1.0 / (3 * predictions.shape[0]) * np.sum(np.power(np.log((predictions + 1)) - np.log((actuals + 1)), 2))) 
 
my_custom_scorer = make_scorer(calculateRMSLE, greater_is_better=True)

# Fit regression model
def predict(X, y, Xtest=None, clf=None, i=0):

    if clf == None:
        if i == 0:
            clf = DecisionTreeRegressor(max_depth=8, random_state=29) #Train RMSLE: 0.532701105932 (normalized) / 0.506345224723 (not norm)
            #clf = GridSearchCV(DecisionTreeRegressor(random_state=29), {'max_depth':[3,8,30]}, cv=10, scoring=my_custom_scorer)
        
        elif i == 1:
            clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=8), n_estimators=100, random_state=29) #best for view so far

        elif i == 2:
            clf = ARDRegression() ### muito muito lento, mas promissor

        elif i == 3:
            clf2 = GradientBoostingRegressor(n_estimators=500, random_state=29) #power ultra foda for view   0.621912412155 (norm) / 0.581863999909 (not norm)
        
        elif i == 4:
            clf = BayesianRidge(copy_X=True)  ### Parece excelente escolha  0.613379421003 (normalized) / 1.10301003297 (not norm)
        
        elif i == 5:
            clf = GridSearchCV(Lasso(), {'alpha':[1,0.01,0.00001]}, cv=10, scoring=my_custom_scorer)
        
        elif i == 6:
            clf = GridSearchCV(LassoLars(), {'alpha':[1,0.01,0.00001]}, cv=10, scoring=my_custom_scorer)
        
        elif i == 7:
            clf = GridSearchCV(MultiTaskLasso(), {'alpha':[1,0.01,0.00001]}, cv=10, scoring=my_custom_scorer)
        
        elif i == 8:
            clf = GridSearchCV(GradientBoostingRegressor(random_state=29), {'n_estimators':[100,300,500,1000]}, cv=10, scoring=my_custom_scorer)

        elif i == 9:
            clf = AdaBoostRegressor(BayesianRidge(), n_estimators=10, random_state=29) #slow, and it is better to use DTR

        elif i == 10:
            clf = PassiveAggressiveRegressor() # Pessimo
        
        elif i == 11:
            clf = LinearRegression()
        
        elif i == 12:
            clf = OrthogonalMatchingPursuit() # bad one
            
        
        print "CLF --- ", clf
        #clf = RidgeCV()
        #clf = Ridge(alpha=)


        #clf = SGDRegressor()  
        #clf = IsotonicRegression()
        # Predict

    if Xtest != None:
        clf.fit(X, y)
        predictions1 = clf.predict(Xtest)
        
        return predictions1

        #print X.shape, Xtest.shape
        newX = np.row_stack((X,Xtest))
        #print newX.shape
        
        print y.shape , predictions1.shape
        newY = np.hstack((y,predictions1))
        print newY.shape

        clf.fit(newX, newY)
        predictions2 = clf.predict(Xtest)

        #predictions = (predictions1 + predictions2)/2.0
        predictions = predictions2
    else:
        #print "Cross val score -- ", np.mean(cross_val_score(clf, X, y, cv=5, n_jobs=-1, scoring=my_custom_scorer))

        #import matplotlib.pyplot as plt
        #res = []
        #for train, test in KFold(len(y), 10):
        #    xtrain, xtest, ytrain, ytest = X[train], X[test], y[train], y[test]
        #    clf.fit(xtrain, ytrain)
        #    yp = clf.predict(xtest)
            #plt.plot(yp, ytest, 'o')
            #plt.plot(ytest, ytest, 'r-')
            #plt.xlim([0,20])
            #plt.ylim([0,20])
        #    res.append( calculateRMSLE(ytest, yp) )
        #print "Manual CV ", np.mean(res)
        #plt.xlabel("Predicted")
        #plt.ylabel("Observed")
        #plt.show()

        clf.fit(X, y)
        #print clf.best_params_
        
        predictions = clf.predict(X)

    predictions[ predictions < 0 ] = 0.0
    #y_2 = clf_2.predict(X)
    return predictions

if trainingStats:

    xvotes_train, xvotes_test, yvotes_train, yvotes_test = train_test_split(X, y_votes, test_size=0.20, random_state=29)
    xcomments_train, xcomments_test, ycomments_train, ycomments_test = train_test_split(X, y_comments, test_size=0.20, random_state=29)
    xviews_train, xviews_test, yviews_train, yviews_test = train_test_split(X, y_views, test_size=0.20, random_state=29)

    for i in range(12):
        print "Iteration ", i

        print "Predicting views  ",
        pred_views = predict(xviews_train, yviews_train,i=i)
        pred_views = predict(xviews_train, yviews_train, xviews_test,i=i)
        #pred_views = predict(xviews_train, yviews_train, clf=GradientBoostingRegressor(n_estimators=500, random_state=29))
        #pred_views = predict(xviews_train, yviews_train, xviews_test, clf=GradientBoostingRegressor(n_estimators=500, random_state=29))
        #pred_views = predict(xviews_train, yviews_train, clf=Lasso(alpha=0.1))
        #pred_views = predict(xviews_train, yviews_train, xviews_test, clf=Lasso(alpha=0.1))

        print "Predicting votes  ",
        pred_votes = predict(xvotes_train, yvotes_train,i=i)
        pred_votes = predict(xvotes_train, yvotes_train, xvotes_test,i=i)

        print "Predicting comments  ",
        pred_comments = predict(xcomments_train, ycomments_train,i=i)
        pred_comments = predict(xcomments_train, ycomments_train, xcomments_test,i=i)

        predictions = np.column_stack((pred_votes, pred_comments, pred_views))
        actuals = np.column_stack((yvotes_test, ycomments_test, yviews_test))

        print "Distributions for training data:"
        print "Views Mean %f, Median %f, Max %f" % (np.mean(y_views), np.median(y_views), np.max(y_views))
        print "Votes Mean %f, Median %f, Max %f" % (np.mean(y_votes), np.median(y_votes), np.max(y_votes))
        print "Comments Mean %f, Median %f, Max %f" % (np.mean(y_comments), np.median(y_comments), np.max(y_comments))

        print "Predictions from the training:"
        print "Views Mean %f, Median %f, Max %f" % (np.mean(pred_views), np.median(pred_views), np.max(pred_views))
        print "Votes Mean %f, Median %f, Max %f" % (np.mean(pred_votes), np.median(pred_votes), np.max(pred_votes))
        print "Comments Mean %f, Median %f, Max %f" % (np.mean(pred_comments), np.median(pred_comments), np.max(pred_comments))

        elements = yvotes_test.shape[0]
        print "Baseline --- all zeros", calculateRMSLE(actuals, np.zeros(3*elements).reshape(elements,3))
        print "Train RMSLE:", calculateRMSLE(actuals, predictions)

    sys.exit(0)

inTest = csv.reader( open("test.csv", "rb") )
header = inTest.next()
testData = [row for row in inTest]

test_descriptions = []
test_summaries = []
test_times = []
test_tags = []
ids = []
test_alltext = []
test_city = []
test_minDists = []
test_sourceL = []

for row in testData:
    #"id","latitude","longitude","summary","description","num_votes","num_comments","num_views","source","created_time","tag_type"
    id, latitude, longitude, summary, description, source, created_time, tag_type = row

    time = datetime.strptime(created_time, "%Y-%m-%d %H:%M:%S")
    timediff = (lastTime - time).days

    test_alltext.append(summary + " " + description + " " + tag_type)
    test_summaries.append(summary)
    test_descriptions.append(description)
    test_times.append(timediff)
    test_tags.append(tag_type)
    ids.append(id)

    #doing something with the latidute and longitude
    cityName, minDist = findClosestCity(float(latitude), float(longitude))
    test_city.append(cityName)
    test_minDists.append(minDist)
    test_sourceL.append(findSource(source))

test_Xcity = ohe.transform(test_city).toarray()
test_Xsource = sohe.transform(test_sourceL).toarray()
test_XminDists = np.array(test_minDists)

dt = vectorizer.transform(test_descriptions).toarray()
st = vectorizer.transform(test_summaries).toarray()
tt = vectorizer.transform(test_tags).toarray()
test_text = tfidf.transform(vectorizer.transform(test_alltext).toarray()).toarray()
test_times = np.array(test_times).astype(int)

#Xtest = np.column_stack((dt,st,tt,test_times))
#Xtest = np.column_stack((test_text,test_times,test_Xcity,test_XminDists))
test_Xsource = sourcepca.transform(test_Xsource)
Xtest = np.column_stack((test_text,test_times,test_Xcity,test_XminDists,test_Xsource))

if normalizeit:
    Xtest = normalize(Xtest)
#Xtest = svd.transform(Xtest)

test_pred_votes = predict(X, y_votes, Xtest)
test_pred_comments = predict(X, y_comments, Xtest)
test_pred_views = predict(X, y_views, Xtest)

print "Predictions from the test:"
print "Views Mean %f, Median %f, Max %f" % (np.mean(test_pred_views), np.median(test_pred_views), np.max(test_pred_views))
print "Votes Mean %f, Median %f, Max %f" % (np.mean(test_pred_votes), np.median(test_pred_votes), np.max(test_pred_votes))
print "Comments Mean %f, Median %f, Max %f" % (np.mean(test_pred_comments), np.median(test_pred_comments), np.max(test_pred_comments))

outfile = open("output.csv", "wb")
open_file_object = csv.writer(outfile)
open_file_object.writerow(["id","num_views","num_votes","num_comments"])
open_file_object.writerows(zip(ids, test_pred_views, test_pred_votes, test_pred_comments))
outfile.close()

