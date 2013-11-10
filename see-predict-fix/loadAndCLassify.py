

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
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import OneHotEncoder, scale, normalize
from sklearn.grid_search import GridSearchCV

#(X,Xtest,y_views,y_comments,y_votes,ids) = pickle.load( open( "save.p", "rb" ) )
(X, Xtest, y_views, y_comments, y_votes, ids, times, Xcity, Xsource, Xtags, test_times, test_Xcity, test_Xsource, test_Xtags) = pickle.load( open( "save.p", "rb" ) )

X = np.column_stack((times,Xcity,Xsource))
Xtest = np.column_stack((test_times, test_Xcity, test_Xsource)) 

clf = GradientBoostingRegressor(n_estimators=50, loss="lad", max_depth=5, alpha=0.99, learning_rate=0.1)
clf.fit(X, y_views)
pred_views = clf.predict(Xtest)

clf.fit(X, y_comments)
pred_comments = clf.predict(Xtest)

clf.fit(X, y_votes)
pred_votes = clf.predict(Xtest)

print "Predictions from the test:"
print "Views Mean %f, Median %f, Max %f" % (np.mean(pred_views), np.median(pred_views), np.max(pred_views))
print "Votes Mean %f, Median %f, Max %f" % (np.mean(pred_votes), np.median(pred_votes), np.max(pred_votes))
print "Comments Mean %f, Median %f, Max %f" % (np.mean(pred_comments), np.median(pred_comments), np.max(pred_comments))

outfile = open("output.csv", "wb")
open_file_object = csv.writer(outfile)
open_file_object.writerow(["id","num_views","num_votes","num_comments"])
open_file_object.writerows(zip(ids, pred_views, pred_votes, pred_comments))
outfile.close()


