
import csv as csv
import numpy as np
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, scale, normalize

inTrain = csv.reader( open("train.csv", "rb") )
header = inTrain.next()
data = [row for row in inTrain]

alltimes = []

for row in data[0:-1:1]:
    id, latitude, longitude, summary, description, num_votes, num_comments, num_views, source, created_time, tag_type = row
    
    time = datetime.strptime(created_time, "%Y-%m-%d %H:%M:%S")
    alltimes.append(time)


weekdays = []
for t in alltimes:
    if t.weekday():
        weekdays.append([0])
    else:
        weekdays.append([1])
week = OneHotEncoder().fit_transform(weekdays).toarray()

hoursl = []
for t in alltimes:
    if t.hour < 6:
        hoursl.append([0])
    elif t.hour < 12:
        hoursl.append([1])
    elif t.hour < 18:
        hoursl.append([2])
    elif t.hour < 24:
        hoursl.append([3])

timeohe = OneHotEncoder()
hours = timeohe.fit_transform(hoursl).toarray()

####  READ TEST:
inTest = csv.reader( open("test.csv", "rb") )
header = inTest.next()
testData = [row for row in inTest]

talltimes = []
thoursl = [ ]
for row in testData:
    id, latitude, longitude, summary, description, source, created_time, tag_type = row
    time = datetime.strptime(created_time, "%Y-%m-%d %H:%M:%S")
    talltimes.append(time)

for t in talltimes:
    if t.hour < 6:
        thoursl.append([0])
    elif t.hour < 12:
        thoursl.append([1])
    elif t.hour < 18:
        thoursl.append([2])
    elif t.hour < 24:
        thoursl.append([3])

test_hours = timeohe.transform(thoursl).toarray()
