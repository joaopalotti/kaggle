from __future__ import division

import numpy as np
import pandas as pd
#Models
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
#auxiliar functions for cross validation
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import make_scorer

from sklearn.preprocessing import StandardScaler

train = pd.read_csv("train.csv.gz", compression="gzip")

y = train["loss"]
del train["loss"]

#simplest way to fill the NA values
train = train.astype(float)
train = train.fillna(train.mean())
X = train.values[:,1:]
X = X.astype(float)

def MAP(actuals, predicted):
    return sum(np.absolute(actuals - predicted)) / actuals.shape[0]
my_map = make_scorer(MAP, greater_is_better=True)

#cross_val_score( LinearRegression(), X, y, scoring=my_map, cv=10)

# preprocessing
preproc = StandardScaler()
X = preproc.fit_transform(X,y)
#Xnorm = normalize(X)

alg = GradientBoostingRegressor()
alg.fit(X,y)


#*******************************************************#
# Whatever was made in the training part, all that it 
# is necessary below is a fitted "alg" for prediction
#*********************  Test  **************************#


test = pd.read_csv("test.csv.gz", compression="gzip")
test = test.astype(float)
#TODO: Should I fill the nan using the training values? 
test = test.fillna(test.mean())
Xtest = test.values[:,1:]
Xtest = Xtest.astype(float)

Xtest = preproc.transform(Xtest)

ids = test.values[:,0]
predicted = alg.predict(Xtest)

outfile = open("output.csv", "wb")
open_file_object = csv.writer(outfile)
pen_file_object.writerow(["id","loss"])
open_file_object.writerows(zip(ids, predicted))
outfile.close()










