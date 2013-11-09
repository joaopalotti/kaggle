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

#TO test:
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSCanonical, PLSRegression
from sklearn.svm import NuSVR, SVR
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor

normalizeit = True
trainingStats = False
onlyOneTrain = False

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
ntags = []
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
sumDes = []

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

def findTag(tag):
    if tag  == 'NA':
        return [0]
    elif tag == 'trash':
        return [1]
    elif tag == 'tree':
        return [2]
    elif tag == 'pothole' or tag == 'traffic':
        return [3]
    elif tag == 'sidewalk':
        return [3]
    elif tag == 'graffiti':
        return [4]
    elif tag == 'street_light':
        return [5]
    elif tag == 'abandoned_vehicles' or tag == "abandoned_vehicle":
        return [6]
    elif tag == 'blighted_property':
        return [7]
    elif tag == 'signs':
        return [8]
    elif tag == 'hydrant' or tag == 'drain_problem' or tag == 'flood':
        return [9]
    elif tag == 'homeless':
        return [10]
    elif tag == 'bike_concern':
        return [11]
    elif tag == 'snow':
        return [12]
    elif tag == 'drug_dealing' or tag == 'robbery':
        return [13]
    else:
        return [99]

#TODO: add this other classes: classes =  ['trash', 'tree', 'pothole', 'graffiti', 'street_light', 'hydrant', 'signs', 'overgrowth', 'sidewalk' , 'blighted_property' , 'traffic' , 'snow' , 'drain_problem' , 'road_safety' , 'bridge' , 'bike_concern' , 'homeless' , 'flood' , 'abandoned_vehicle' , 'abandoned_vehicles' , 'crosswalk' , 'drug_dealing' , 'robbery' , 'parking_meter' , 'bench' , 'animal_problem' , 'odor' , 'noise_complaint' , 'test' , 'illegal_idling' , 'street_signal' , 'rodents' , 'heat' , 'prostitution' , 'roadkill' , 'bad_driving' , 'pedestrian_light' , 'zoning' , 'lost_and_found' , 'public_art' , 'public_concern' , 'other']


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
    ntags.append(tag_type)
    tags.append(findTag(tag_type))

    sumDes.append(summary + " " + description) 
    alltext.append(summary + " " + description + " " + tag_type)
    summaries.append(summary)
    descriptions.append(description)
    
    # lattitude and longitude measures
    cityName, minDist = findClosestCity(float(latitude), float(longitude))
    city.append(cityName)
    minDists.append(minDist)
    sourceL.append(findSource(source))
ntags = np.array(ntags)
sumDes = np.array(sumDes)


####  READ TEST:
inTest = csv.reader( open("test.csv", "rb") )
header = inTest.next()
testData = [row for row in inTest]

test_descriptions = []
test_summaries = []
test_times = []
test_ntags = []
test_tags = []
ids = []
test_alltext = []
test_city = []
test_minDists = []
test_sourceL = []
test_sumDes = []

for row in testData:
    #"id","latitude","longitude","summary","description","num_votes","num_comments","num_views","source","created_time","tag_type"
    id, latitude, longitude, summary, description, source, created_time, tag_type = row

    time = datetime.strptime(created_time, "%Y-%m-%d %H:%M:%S")
    timediff = (lastTime - time).days

    test_alltext.append(summary + " " + description + " " + tag_type)
    test_sumDes.append(summary + " " + description)
    test_summaries.append(summary)
    test_descriptions.append(description)
    test_times.append(timediff)
    test_tags.append(findTag(tag_type))
    test_ntags.append(tag_type)
    ids.append(id)

    #doing something with the latidute and longitude
    cityName, minDist = findClosestCity(float(latitude), float(longitude))
    test_city.append(cityName)
    test_minDists.append(minDist)
    test_sourceL.append(findSource(source))
test_ntags = np.array(test_ntags)
test_sumDes = np.array(test_sumDes)


def classify(sumDes, ntags, t_sumDes, t_ntags, subject):

    xna = sumDes[ ntags == 'NA']
    xthing = sumDes[ntags == subject]
    xnthing = sumDes[ (ntags != 'NA') & (ntags != subject)]

    t_xna = t_sumDes [ t_ntags == 'NA' ]
    t_xthing = t_sumDes[ t_ntags == subject]
    t_xnthing = t_sumDes[ (t_ntags != 'NA') & (t_ntags != subject)]

    X = np.hstack((xthing, xnthing, t_xthing, t_xnthing))
    y = np.hstack((np.ones(xthing.shape[0]), np.zeros(xnthing.shape[0]), np.ones(t_xthing.shape[0]), np.zeros(t_xnthing.shape[0])))

    vectorizer = CountVectorizer(max_features=100, stop_words="english", strip_accents="ascii")
    X = vectorizer.fit_transform(X).toarray()
    xna = vectorizer.transform(xna).toarray()
    t_xna = vectorizer.transform(t_xna).toarray()

    from sklearn.ensemble import ExtraTreesClassifier
    pred = (ExtraTreesClassifier().fit(X,y).predict_proba(xna)[:,1] > 0.95).astype(str)
    t_pred = (ExtraTreesClassifier().fit(X,y).predict_proba(t_xna)[:,1] > 0.95).astype(str)
    
    print "Transformed ---> ", pred[pred == "True"].shape[0]
    pred[ pred == 'True'] = subject
    t_pred[ t_pred == 'True'] = subject
    pred[ pred == 'False' ] = 'NA'
    t_pred[ t_pred == 'False' ] = 'NA'

    ntags[ ntags == 'NA'] = pred
    t_ntags[ t_ntags == 'NA'] = t_pred
    
    return ntags, t_ntags


classes =  ['trash', 'tree', 'pothole', 'graffiti', 'street_light', 'hydrant', 'signs', 'overgrowth', 'sidewalk' , 'blighted_property' , 'traffic' , 'snow' , 'drain_problem' , 'road_safety' , 'bridge' , 'bike_concern' , 'homeless' , 'flood' , 'abandoned_vehicle' , 'abandoned_vehicles' , 'crosswalk' , 'drug_dealing' , 'robbery' , 'parking_meter' , 'bench' , 'animal_problem' , 'odor' , 'noise_complaint' , 'test' , 'illegal_idling' , 'street_signal' , 'rodents' , 'heat' , 'prostitution' , 'roadkill' , 'bad_driving' , 'pedestrian_light' , 'zoning' , 'lost_and_found' , 'public_art' , 'public_concern' , 'other']

print "Repeating 5 times:"
for x in range(5):
    for c in classes:
        print "Tag --> ", c
        print "#NA ---> ", ntags[ntags == 'NA'].shape[0]
        print "t_#NA ---> ", test_ntags[test_ntags == 'NA'].shape[0]
        ntags, test_ntags = classify(sumDes, ntags, test_sumDes, test_ntags, c)

tags = []
for tag in ntags:
    tags.append(findTag(tag))

ohe = OneHotEncoder()
sohe = OneHotEncoder()
tohe = OneHotEncoder()
Xcity = ohe.fit_transform(city).toarray()
Xsource = sohe.fit_transform(sourceL).toarray()
print "Vectorizing"
XminDists = np.array(minDists)

#vectorizer = HashingVectorizer(n_features=100, strip_accents="ascii")  # training: 0.971392246367
#vectorizer = HashingVectorizer(n_features=1000, strip_accents="ascii")
#vectorizer = HashingVectorizer(n_features=2000, strip_accents="ascii")
#vectorizer = CountVectorizer(max_features=100, stop_words="english", strip_accents="ascii")#, analyzer='char_wb') #seems good for big data
#tfidf = TfidfTransformer()

#dt = vectorizer.fit_transform(descriptions).toarray()
#st = vectorizer.fit_transform(summaries).toarray()
#tt = vectorizer.fit_transform(tags).toarray()
#text = vectorizer.fit_transform(alltext).toarray()
#text = vectorizer.fit_transform(sumDes).toarray()
Xtags = tohe.fit_transform(tags).toarray()

#text = tfidf.fit_transform(vectorizer.fit_transform(alltext).toarray()).toarray()

times = np.array(times).astype(int)
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

X = np.column_stack((times,Xcity,Xsource,Xtags))
#X = np.column_stack((text,times,Xcity,XminDists,Xsource,Xtags))
#X = np.column_stack((text,times,Xcity,XminDists,Xsource))
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

xxxxx

# Fit regression model
def predict(X, y, Xtest=None, clf=None, i=-1):

    if clf == None:

        clf = GradientBoostingRegressor(n_estimators=50, loss="lad", max_depth=5, alpha=0.99, learning_rate=0.1)

        if i == 0:
            clf = DecisionTreeRegressor(max_depth=8, random_state=29) #Train RMSLE: 0.532701105932 (normalized) / 0.506345224723 (not norm)
            #clf = GridSearchCV(DecisionTreeRegressor(random_state=29), {'max_depth':[3,8,30]}, cv=10, scoring=my_custom_scorer)
        
        elif i == 1:
            clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=8), n_estimators=100, random_state=29) #best for view so far

        elif i == 2:
            #clf = ARDRegression() ### muito muito lento, mas promissor
            clf = DecisionTreeRegressor(max_depth=3, random_state=29) #Train RMSLE: 0.532701105932 (normalized) / 0.506345224723 (not norm)

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
            
        elif i == 99:
            clf = DecisionTreeRegressor(max_depth=8, random_state=29) #Train RMSLE: 0.532701105932 (normalized) / 0.506345224723 (not norm)
        
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
        
        print y.shape, predictions1.shape
        newY = np.hstack((y,predictions1))
        print newY.shape

        clf.fit(newX, newY)
        predictions2 = clf.predict(Xtest)

        #predictions = (predictions1 + predictions2)/2.0
        predictions = predictions2
    else:
        print "Cross val score -- ", np.mean(cross_val_score(clf, X, y, cv=5, n_jobs=-1, scoring=my_custom_scorer))

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

if onlyOneTrain:
    predict(X, y_views, i=99)
    sys.exit(0)

if trainingStats:

    for i in range(12):
        print "Iteration ", i
        results = [] 
        for j in range(5):
        
            xvotes_train, xvotes_test, yvotes_train, yvotes_test = train_test_split(X, y_votes, test_size=0.20, random_state=j)
            xcomments_train, xcomments_test, ycomments_train, ycomments_test = train_test_split(X, y_comments, test_size=0.20, random_state=j)
            xviews_train, xviews_test, yviews_train, yviews_test = train_test_split(X, y_views, test_size=0.20, random_state=j)

            #print "Predicting views  ",
            #pred_views = predict(xviews_train, yviews_train,i=i)
            pred_views = predict(xviews_train, yviews_train, xviews_test,i=i)
            #pred_views = predict(xviews_train, yviews_train, clf=GradientBoostingRegressor(n_estimators=500, random_state=29))
            #pred_views = predict(xviews_train, yviews_train, xviews_test, clf=GradientBoostingRegressor(n_estimators=500, random_state=29))
            #pred_views = predict(xviews_train, yviews_train, clf=Lasso(alpha=0.1))
            #pred_views = predict(xviews_train, yviews_train, xviews_test, clf=Lasso(alpha=0.1))

            #print "Predicting votes  ",
            #pred_votes = predict(xvotes_train, yvotes_train,i=i)
            pred_votes = predict(xvotes_train, yvotes_train, xvotes_test,i=i)

            #print "Predicting comments  ",
            #pred_comments = predict(xcomments_train, ycomments_train,i=i)
            pred_comments = predict(xcomments_train, ycomments_train, xcomments_test,i=i)

            predictions = np.column_stack((pred_votes, pred_comments, pred_views))
            actuals = np.column_stack((yvotes_test, ycomments_test, yviews_test))

            #print "Distributions for training data:"
            #print "Views Mean %f, Median %f, Max %f" % (np.mean(y_views), np.median(y_views), np.max(y_views))
            #print "Votes Mean %f, Median %f, Max %f" % (np.mean(y_votes), np.median(y_votes), np.max(y_votes))
            #print "Comments Mean %f, Median %f, Max %f" % (np.mean(y_comments), np.median(y_comments), np.max(y_comments))

            #print "Predictions from the training:"
            #print "Views Mean %f, Median %f, Max %f" % (np.mean(pred_views), np.median(pred_views), np.max(pred_views))
            #print "Votes Mean %f, Median %f, Max %f" % (np.mean(pred_votes), np.median(pred_votes), np.max(pred_votes))
            #print "Comments Mean %f, Median %f, Max %f" % (np.mean(pred_comments), np.median(pred_comments), np.max(pred_comments))

            elements = yvotes_test.shape[0]
            #print "Baseline --- all zeros", calculateRMSLE(actuals, np.zeros(3*elements).reshape(elements,3))
            #print "Train RMSLE:",
            results.append(calculateRMSLE(actuals, predictions))
        print "RMSLE final ---> ", np.mean(results)

    sys.exit(0)



###### TEST:

for c in classes:
    print "Tag --> ", c
    print "#NA ---> ", ntags[ntags == 'NA'].shape[0]
    ntags = classify(sumDes, ntags, c)

test_tags = []
for tag in test_ntags:
    test_tags.append(findTag(tag))

test_Xcity = ohe.transform(test_city).toarray()
test_Xsource = sohe.transform(test_sourceL).toarray()
#test_XminDists = np.array(test_minDists)

#dt = vectorizer.transform(test_descriptions).toarray()
#st = vectorizer.transform(test_summaries).toarray()
#tt = vectorizer.transform(test_tags).toarray()
#test_text = tfidf.transform(vectorizer.transform(test_alltext).toarray()).toarray()
#test_text = vectorizer.transform(test_sumDes).toarray()
test_Xtags = tohe.transform(test_tags).toarray()
test_times = np.array(test_times).astype(int)

#Xtest = np.column_stack((dt,st,tt,test_times))
#Xtest = np.column_stack((test_text,test_times,test_Xcity,test_XminDists))
test_Xsource = sourcepca.transform(test_Xsource)
Xtest = np.column_stack((test_times,test_Xcity,test_Xsource, test_Xtags))
#Xtest = np.column_stack((test_text,test_times,test_Xcity,test_XminDists,test_Xsource, test_Xtags))

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

