
from __future__ import division
import math
#from scoop import futures
import gzip, sys
from collections import defaultdict 

#content = gzip.open("test.gz", "rb")
content = gzip.open("yandex_train.gz", "rb")
#content = gzip.open("smallyandex.gz", "rb")
urlDomain = {}
docPopularity = defaultdict(int)
domainPopularity = defaultdict(int)
docClickedPopularity = defaultdict(int)
domainClickedPopularity = defaultdict(int)

savedTimes = defaultdict(int)
docClass = defaultdict(int)

def idcg(rel):
    return dcg(rel) / dcg(sorted(rel, reversed=True))

def dcg(rel):
    return sum([((2**rel[i-1])-1)/(math.log(i+1,2)) for i in range(1,len(rel)+1)])

def calculateRelevanceScores(bla):
    return

def readOneLine(line):

    field = line.strip().split("\t")
    sessionId = field[0]
    secondField = field[1]
    urls = []
    lastActionClick = False

    if secondField == "M":
        typeOfRecord = "M"
        day = field[2] 
        userid = field[3]

    else:
        timePassed = field[1]
        typeOfRecord = field[2]

        if typeOfRecord == "Q" or typeOfRecord == "T":
            serpid = field[3]
            savedTimes[serpid] = timePassed

            queryid = field[4]
            terms = field[5]
            urls = field[6:]
            urldomain = [u.split(",") for u in urls]
            for (url, domain) in urldomain:
                docPopularity[url] += 1
                domainPopularity[domain] += 1
                urlDomain[url] = domain
            
            #if typeOfRecord == "T":
            #    print "T"

        elif typeOfRecord == "C":
            
            lastActionClick = True
            serpid = field[3]
            urlid = field[4]
            docClickedPopularity[urlid] += 1
            domainClickedPopularity[ urlDomain[urlid] ] += 1

            dwellTime = int(timePassed) - int(savedTimes[serpid])
            savedTimes[serpid] = timePassed

            dclass = 0
            if dwellTime >= 50 and dwellTime <= 399:
                dclass = 1
            elif dwellTime >= 400:
                dclass = 2
            
            if docClass[urlid] < dclass:
                docClass[urlid] = dclass

    if lastActionClick:
        return urlid
    else:
        return -1

#futures.map(readOneLine, content)
#map(readOneLine, content)
for line in content:
    urlid = readOneLine(line)
    if urlid > 0 and docClass[urlid] < 2:
        docClass[urlid] = 2

for docid in urlDomain.keys():
    print docid, docClass[docid], docPopularity[docid], domainPopularity[urlDomain[docid] ], docClickedPopularity[docid], domainClickedPopularity[urlDomain[docid]]


### Baseline



