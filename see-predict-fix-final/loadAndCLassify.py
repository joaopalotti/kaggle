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
from numpy.random import random_sample
from sklearn.svm import SVR

#(X,Xtest,y_views,y_comments,y_votes,ids) = pickle.load( open( "save.p", "rb" ) )
#(X, Xtest, y_views, y_comments, y_votes, ids, times, Xcity, Xsource, Xtags, test_times, test_Xcity, test_Xsource, test_Xtags) 
(X, Xtest, y_views, y_comments, y_votes, ids, times, hours, Xcity, Xsource, Xtags, h, months, lati, longi, times2, months2, test_times, test_Xcity, test_Xsource, test_Xtags, test_hours, test_h, test_months, test_lati, test_longi, test_times2, test_months2) = pickle.load( open( "save.p", "rb" ) )


#smallX = np.column_stack((times,Xcity,Xsource))
#bigX = np.column_stack((times,Xcity,Xsource,lati,longi,h,months))
#X = np.column_stack((times,Xcity,Xsource, h, months, lati, longi, Xtags))
X = np.column_stack((times2, Xcity, Xsource, h, months2, lati, longi, Xtags))
Xtest = np.column_stack((test_times2, test_Xcity, test_Xsource, test_h, test_months2, test_lati, test_longi, test_Xtags))

newy_views = np.log(y_views + 1)
mi_vi = newy_views.min()
ma_vi = newy_views.max()
nviews = (newy_views - mi_vi) / (ma_vi - mi_vi)

newy_comments = np.log(y_comments + 1)
mi_co = newy_comments.min()
ma_co = newy_comments.max()
ncomments = (newy_comments - mi_co) / (ma_co - mi_co)

newy_votes = np.log(y_votes + 1)
mi_vt = newy_votes.min()
ma_vt = newy_votes.max()
nvotes = (newy_votes - mi_vt) / (ma_vt - mi_vt)


#s = random_sample(X.shape[0])
#X = X[s>0.90]
#newy_views = newy_views[s>0.90]
#newy_votes = newy_votes[s>0.90]
#newy_comments = newy_comments[s>0.90]


###### views:
#clf1 = ExtraTreesRegressor(n_estimators=50, max_depth=8)
#clf1.fit(X, newy_views)
#pred_views1 = clf1.predict(Xtest)

#newX = np.row_stack((X, Xtest))
#newY = np.hstack((newy_views, pred_views1))

#clf2 = GradientBoostingRegressor(n_estimators=50, loss="huber", max_depth=6) #subsample=0.2)
#pred_views = clf2.fit(newX, newY).predict(Xtest)
#pred_views = clf2.fit(X, newy_views).predict(Xtest)
#pred_views = np.exp(pred_views) - 1
#pred_views[ pred_views < 0 ] = 0.0

# new approach using SVM:
#etr = ExtraTreesRegressor(n_estimators=50, max_depth=8)
#newScaledX  = etr.fit_transform(scale(X), newy_views)
#newTestScaledX  = etr.transform(scale(Xtest))
#svr = SVR(max_iter=10000000, C=1.001, kernel="linear", epsilon=0.2)

#pred_views1 = svr.fit(newScaledX, newy_views).predict(newTestScaledX)
#pred_views2 = LinearRegression().fit(X,newy_views).predict(Xtest)
#pred_views3 = Ridge(normalize=False, solver="auto").fit(X,newy_views).predict(Xtest)

pred_views1 = GradientBoostingRegressor(n_estimators=50, loss="huber", max_depth=6, random_state=1).fit(X,nviews).predict(Xtest)
pred_views2 = GradientBoostingRegressor(n_estimators=50, loss="huber", max_depth=6, random_state=2).fit(X,nviews).predict(Xtest) 
pred_views3 = GradientBoostingRegressor(n_estimators=50, loss="huber", max_depth=6, random_state=3).fit(X,nviews).predict(Xtest)

pred_views = (pred_views1 + pred_views2 + pred_views3) / 3.0
pred_views = (pred_views * (ma_vi - mi_vi)) + mi_vi

pred_views = np.exp(pred_views) - 1
pred_views[ pred_views < 0 ] = 0.0


###### comments:
#clf1 = ExtraTreesRegressor(n_estimators=50, max_depth=8)
#clf1.fit(X, newy_comments)
#pred_comments1 = clf1.predict(Xtest)

#newX = np.row_stack((X, Xtest))
#newY = np.hstack((newy_comments, pred_comments1))

clf2 = GradientBoostingRegressor(n_estimators=30, loss="ls", max_depth=7)
pred_comments = clf2.fit(X, ncomments).predict(Xtest)
#pred_comments = clf2.fit(newX, newY).predict(Xtest)

pred_comments = (pred_comments * (ma_co - mi_co)) + mi_co
pred_comments = np.exp(pred_comments) - 1
pred_comments[ pred_comments < 0 ] = 0.0

###### votes:
#clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=11), n_estimators=10, learning_rate=0.01, loss="square" )
#clf1 = ExtraTreesRegressor(n_estimators=30, max_depth=11)
#clf1.fit(X, newy_votes)
#pred_votes1 = clf1.predict(Xtest)

#newX = np.row_stack((X, Xtest))
#newY = np.hstack((newy_votes, pred_votes1))

clf2 = GradientBoostingRegressor(n_estimators=50, loss="ls", max_depth=6)
pred_votes = clf2.fit(X, nvotes).predict(Xtest)
#pred_votes = clf2.fit(newX, newY).predict(Xtest)
pred_votes = (pred_votes * (ma_vt - mi_vt)) + mi_vt
pred_votes = np.exp(pred_votes) - 1
pred_votes[ pred_votes < 0 ] = 0.0


print "Predictions from the test:"
print "Views Mean %f, Median %f, Max %f" % (np.mean(pred_views), np.median(pred_views), np.max(pred_views))
print "Votes Mean %f, Median %f, Max %f" % (np.mean(pred_votes), np.median(pred_votes), np.max(pred_votes))
print "Comments Mean %f, Median %f, Max %f" % (np.mean(pred_comments), np.median(pred_comments), np.max(pred_comments))

outfile = open("output.csv", "wb")
open_file_object = csv.writer(outfile)
open_file_object.writerow(["id","num_views","num_votes","num_comments"])
open_file_object.writerows(zip(ids, pred_views, pred_votes, pred_comments))
outfile.close()


