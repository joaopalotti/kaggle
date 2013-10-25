import csv as csv
import math
from collections import Counter
import numpy as np
from datetime import datetime

from deap import algorithms, base, creator, tools, gp

inTrain = csv.reader( open("train.csv", "rb") )
header = inTrain.next()
data = [row for row in inTrain]

goal = []
views = []
votes = []
comments = []
times = []

lastTime = datetime.strptime("2013-04-30 23:51:37", "%Y-%m-%d %H:%M:%S")

transformedInput = []

for row in data:
    #"id","latitude","longitude","summary","description","num_votes","num_comments","num_views","source","created_time","tag_type"
    id, latitude, longitude, summary, description, num_votes, num_comments, num_views, source, created_time, tag_type = row

    time = datetime.strptime(created_time, "%Y-%m-%d %H:%M:%S")

    votes.append(num_votes)
    comments.append(num_comments)
    views.append(num_views)
    times.append(time)

    wordsInSummary = len(summary.split())
    charsInSummary = len(summary)
    wordsInDescription = len(description.split())
    charsInDescription = len(description)
    
    days = (lastTime - time).days
    transformedInput.append( [wordsInSummary, charsInSummary, wordsInDescription, charsInDescription, days, num_votes, num_comments, num_views] )
    #goal.append( map(int, [num_votes, num_comments, num_views]) )

csvout = csv.writer(open('transformed.csv', 'wb'))
csvout.writerows(transformedInput)


#cview = Counter(views)
#ccomments = Counter(comments)
#cvotes = Counter(votes)

# print min and max values
#print "Views - ", min( map(int,cview.keys()) ), max( map(int,cview.keys()))
#print "Comments - ", min( map(int,ccomments.keys()) ), max( map(int,ccomments.keys()))
#print "Votes - ", min( map(int,cvotes.keys()) ), max( map(int,cvotes.keys()))
#print "Times - ", min(times), max(times)

numExamples = len(goal)
supersum = 0.0
for example in goal:
    predict = 0
    supersum += math.pow( math.log(predict + 1) - math.log(example[0] + 1), 2 )
    supersum += math.pow( math.log(predict + 1) - math.log(example[1] + 1), 2)
    supersum += math.pow( math.log(predict + 1) - math.log(example[2] + 1), 2)
rmsle = math.sqrt( 1.0/(numExamples*3) * supersum )
print rmsle

inTest = csv.reader( open("test.csv", "rb") )
testHeader = inTest.next()
testData = [row for row in inTest]
csvout = csv.writer(open('output.csv', 'wb'))

for row in testData:
    id = row[0]
    csvout.writerow([id,0,0,0])



