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

def calculateRMSLE(actuals, predictions):
    predictions[ predictions < 0 ] = 0.0
    return math.sqrt( 1.0 / (3 * predictions.shape[0]) * np.sum(np.power(np.log((predictions + 1)) - np.log((actuals + 1)), 2)))
my_custom_scorer = make_scorer(calculateRMSLE, greater_is_better=True)

def calculateRMSLE2(actuals, predictions):
    predictions = np.exp(predictions) - 1
    actuals = np.exp(actuals) - 1
    predictions[ predictions < 0 ] = 0.0
    return math.sqrt( 1.0 / (3 * predictions.shape[0]) * np.sum(np.power(np.log((predictions + 1)) - np.log((actuals + 1)), 2)))
my_custom_scorer2 = make_scorer(calculateRMSLE2, greater_is_better=True)

def hoursplit(allt):
    hoursl = []
    for t in allt:
        hoursl.append([int(t.hour)])
    return np.array(hoursl)

def monthspassed(allt):
    #lastTime = datetime.strptime("2013-05-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    lastTime = datetime.strptime("2013-10-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    dlist = []
    for t in allt:
        dlist.append( (lastTime - t).days / 7 )
    return np.array(dlist)

def timesplit(t):
    if t < 50:
        return [0]
    elif t < 100:
        return [1]
    elif t< 150:
        return [2]
    elif t<200:
        return [3]
    elif t < 250:
        return [4]
    elif t< 300:
        return [5]
    elif t < 350:
        return [6]
    elif t<400:
        return [7]
    elif t < 450:
        return [8]
    else:
        return [9]
