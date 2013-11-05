import csv as csv
import math
from collections import Counter
import numpy as np
from datetime import datetime

from deap import algorithms, base, creator, tools, gp
from sklearn.cross_validation import cross_val_score, KFold

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

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

lastTime = datetime.strptime("2013-04-30 23:51:37", "%Y-%m-%d %H:%M:%S")

for row in data:
    #"id","latitude","longitude","summary","description","num_votes","num_comments","num_views","source","created_time","tag_type"
    id, latitude, longitude, summary, description, num_votes, num_comments, num_views, source, created_time, tag_type = row

    time = datetime.strptime(created_time, "%Y-%m-%d %H:%M:%S")

    votes.append(num_votes)
    comments.append(num_comments)
    views.append(num_views)
    times.append(time)

    #TODO: do something with the latidute and longitude

    summaries.append(summary)
    descriptions.append(description)

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
vectorizer = HashingVectorizer(n_features=100, strip_accents="ascii")
#vectorizer = TfidfTransformer()

dt = vectorizer.fit_transform(descriptions).toarray()
st = vectorizer.fit_transform(summaries).toarray()

X = np.hstack((dt,st))
y_votes = np.array(votes).astype(int)
y_comments = np.array(comments).astype(int)
y_views = np.array(views).astype(int)


def calculateRMSLE(actuals, predictions, nelements):
    return math.sqrt( 1.0 / nelements * np.sum(np.power(np.log((predictions + 1)) - np.log((actuals + 1)), 2))) 
 

# Fit regression model
def predict(X, y, Xtest=None):

    clf = DecisionTreeRegressor(max_depth=4)
    #clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=29)


    # Predict
    if Xtest != None:
        clf.fit(X, y)
        predictions = clf.predict(Xtest)
    else:
        #print "Cross val score -- ", np.mean(cross_val_score(clf, X, y, cv=10, n_jobs=-1))

        import matplotlib.pyplot as plt
        
        res = []
        for train, test in KFold(len(y), 10):
            xtrain, xtest, ytrain, ytest = X[train], X[test], y[train], y[test]

            clf.fit(xtrain, ytrain)
            yp = clf.predict(xtest)
            
            plt.plot(yp, ytest, 'o')
            plt.plot(ytest, ytest, 'r-')
            plt.xlim([0,20])
            plt.ylim([0,20])
            
            res.append( calculateRMSLE(ytest, yp, ytest.shape[0]) )
        
        print "Manual CV ", np.mean(res)
        plt.xlabel("Predicted")
        plt.ylabel("Observed")
        plt.show()

        clf.fit(X, y)
        predictions = clf.predict(X)

    #y_2 = clf_2.predict(X)
    return predictions

pred_votes = predict(X, y_votes)
pred_comments = predict(X, y_comments)
pred_views = predict(X, y_views)

predictions = np.vstack((pred_votes, pred_comments, pred_views))
actuals = np.vstack((y_votes, y_comments, y_views))

elements = y_votes.shape[0]
print calculateRMSLE(actuals, predictions, actuals.shape[1])
print "Baseline --- all zeros", calculateRMSLE(actuals, np.zeros(3*elements).reshape(3,elements),  actuals.shape[1] )

inTest = csv.reader( open("test.csv", "rb") )
header = inTest.next()
testData = [row for row in inTest]

test_descriptions = []
test_summaries = []

ids = []
for row in testData:
    #"id","latitude","longitude","summary","description","num_votes","num_comments","num_views","source","created_time","tag_type"
    id, latitude, longitude, summary, description, source, created_time, tag_type = row

    #time = datetime.strptime(created_time, "%Y-%m-%d %H:%M:%S")

    #TODO: do something with the latidute and longitude

    test_summaries.append(summary)
    test_descriptions.append(description)
    ids.append(id)

dt = vectorizer.fit_transform(descriptions).toarray()
st = vectorizer.fit_transform(summaries).toarray()

Xtest = np.hstack((dt,st))

test_pred_votes = predict(X, y_votes, Xtest)
test_pred_comments = predict(X, y_comments, Xtest)
test_pred_views = predict(X, y_views, Xtest)


outfile = open("output.csv", "wb")
open_file_object = csv.writer(outfile)
open_file_object.writerow(["id","num_views","num_votes","num_comments"])
open_file_object.writerows(zip(ids, test_pred_views, test_pred_votes, test_pred_comments))
outfile.close()


