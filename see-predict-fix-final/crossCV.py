import pickle
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
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
from sklearn.preprocessing import OneHotEncoder, scale, normalize
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor

def calculateRMSLE2(actuals, predictions):
    predictions = np.exp(predictions) - 1
    actuals = np.exp(actuals) - 1
    predictions[ predictions < 0 ] = 0.0
    return math.sqrt( 1.0 / (3 * predictions.shape[0]) * np.sum(np.power(np.log((predictions + 1)) - np.log((actuals + 1)), 2)))
my_custom_scorer2 = make_scorer(calculateRMSLE2, greater_is_better=True)

def calculateRMSLE3(actuals, predictions):
    predictions = np.exp(predictions) - 1.1
    actuals = np.exp(actuals) - 1
    predictions[ predictions < 0 ] = 0.0
    return math.sqrt( 1.0 / (3 * predictions.shape[0]) * np.sum(np.power(np.log((predictions + 1)) - np.log((actuals + 1)), 2)))
my_custom_scorer3 = make_scorer(calculateRMSLE3, greater_is_better=True)



(X, Xtest, y_views, y_comments, y_votes, ids, times, hours, Xcity, Xsource, Xtags, h, months, lati, longi, test_times, test_Xcity, test_Xsource, test_Xtags, test_hours, test_h, test_months, test_lati, test_longi) = pickle.load( open( "save.p", "rb" ) )

smallX = np.column_stack((times,Xcity,Xsource))
bigX = np.column_stack((times,Xcity,Xsource,lati,longi,h,months))
X = np.column_stack((times,Xcity,Xsource, h, months, lati, longi, Xtags))

newy_views = np.log(y_views + 1)
newy_votes = np.log(y_votes + 1)
newy_comments = np.log(y_comments + 1)

kf = KFold( X.shape[0], n_folds=10, random_state=0)
i = 0

views1, views2, viewsm, comments1,comments2, commentsm, votes1,votes2, votesm = [],[],[],[],[],[],[],[],[]

for train_index, test_index in kf:
    i += 1
    print "Loop ", i

    X_train, X_test = X[train_index], X[test_index]
    yviews_train, yviews_test = newy_views[train_index], newy_views[test_index]
    ycomments_train, ycomments_test = newy_comments[train_index], newy_comments[test_index]
    yvotes_train, yvotes_test = newy_votes[train_index], newy_votes[test_index]

    # views:
    clf1 = ExtraTreesRegressor(n_estimators=10, max_depth=9, random_state=0)
    pred_views1 = clf1.fit(X_train, yviews_train).predict(X_test)
    v1 = calculateRMSLE3(pred_views1, yviews_test)
    print "Views clf1 = ", v1

    newX = np.row_stack((X_train, X_test))
    newY = np.hstack((yviews_train, pred_views1))

    clf2 = GradientBoostingRegressor(n_estimators=50, loss="huber", max_depth=6, random_state=0) #subsample=0.2)
    #pred_views2 = clf2.fit(X_train, yviews_train).predict(X_test)
    pred_views2 = clf2.fit(newX, newY).predict(X_test)
    v2 = calculateRMSLE3(pred_views2, yviews_test)
    print "Views clf2 = ", v2
    
    vm = calculateRMSLE2( ((pred_views1 + pred_views2) / 2.0), yviews_test)
    print "Merged version = ",  vm

    views1.append(v1)
    views2.append(v2)
    viewsm.append(vm)
    
    #comments
    #clf1 = ExtraTreesRegressor(n_estimators=50, max_depth=8)
    #pred_comments1 = clf1.fit(X_train, ycomments_train).predict(X_test)
    #c1 = calculateRMSLE2(pred_comments1, ycomments_test)
    #print "Comments clf1 = ", c1
    
    #newY = np.hstack((ycomments_train, pred_comments1))

    #clf2 = GradientBoostingRegressor(n_estimators=30, loss="ls", max_depth=7)
    #pred_comments2 = clf2.fit(X_train, ycomments_train).predict(X_test)
    #pred_comments2 = clf2.fit(newX, newY).predict(X_test)
    #c2 = calculateRMSLE2(pred_comments2, ycomments_test)
    #print "Comments clf2 = ", c2

    #cm = calculateRMSLE2( ((pred_comments1 + pred_comments2) / 2.0), ycomments_test)
    #print "Merged version = ",  cm
    #comments1.append(c1)
    #comments2.append(c2)
    #commentsm.append(cm)
    
    #votes
    #clf1 = ExtraTreesRegressor(n_estimators=30, max_depth=11)
    #pred_votes1 = clf1.fit(X_train, yvotes_train).predict(X_test)
    #vo1 = calculateRMSLE2(pred_votes1, yvotes_test)
    #print "Votes clf1 = ", vo1
    
    #newY = np.hstack((yvotes_train, pred_votes1))

    #clf2 = GradientBoostingRegressor(n_estimators=50, loss="ls", max_depth=6)
    #pred_votes2 = clf2.fit(X_train, yvotes_train).predict(X_test)
    #pred_votes2 = clf2.fit(newX, newY).predict(X_test)
    #vo2 = calculateRMSLE2(pred_votes2, yvotes_test)
    #print "Votes clf2 = ", vo2

    #vom = calculateRMSLE2( ((pred_votes1 + pred_votes2) / 2.0), yvotes_test)
    #print "Merged version = ", vom

    #votes1.append(vo1)
    #votes2.append(vo2)
    #votesm.append(vom)


v1, v2, vm = np.array(views1), np.array(views2), np.array(viewsm)
print "Views "
print "Model 1: ", np.mean(v1), np.std(v1), np.min(v1), np.max(v1)
print "Model 2: ", np.mean(v2), np.std(v2), np.min(v2), np.max(v2)
print "Model M: ", np.mean(vm), np.std(vm), np.min(vm), np.max(vm)

#c1, c2, cm = np.array(comments1), np.array(comments2), np.array(commentsm)
#print "Comments " 
#print "Model 1: ", np.mean(c1), np.std(c1), np.min(c1), np.max(c1)
#print "Model 2: ", np.mean(c2), np.std(c2), np.min(c2), np.max(c2)
#print "Model M: ", np.mean(cm), np.std(cm), np.min(cm), np.max(cm)

#vo1, vo2, vom = np.array(votes1), np.array(votes2), np.array(votesm)

#print "Votes "
#print "Model 1: ", np.mean(vo1), np.std(vo1), np.min(vo1), np.max(vo1)
#print "Model 2: ", np.mean(vo2), np.std(vo2), np.min(vo2), np.max(vo2)
#print "Model M: ", np.mean(vom), np.std(vom), np.min(vom), np.max(vom)


