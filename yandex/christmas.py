from __future__ import print_function
from __future__ import division
import pandas as pd
from collections import defaultdict
import gzip, csv, math
import numpy as np
import matplotlib.pyplot as plt
import ast
import sys

#inputfile = "smallyandex.gz"
#inputfile = "mediumyandex.gz"
inputfile = "all.gz"

filetouse= "stable%d.csv"
#filetouse= "small%d.csv"

def stable1(file, outname="stable1.csv"):
    content = gzip.open(file, "rb")
    fout = open(outname, "wb")
    out = csv.writer(fout)

    out.writerow(["SessionId", "Type", "Day", "UserId"])
    for line in content:
        fields = line.strip().split("\t")
        sessionId = fields[0]
        second = fields[1]

        if second == "M":
            day = fields[2]
            userId = fields[3]
        else:
            continue
        out.writerow([sessionId, second, day, userId])
    fout.close()

def stable2(file, outname="stable2.csv"):
    content = gzip.open(file, "rb")
    fout = open(outname, "wb")
    out = csv.writer(fout)

    #SessionID TimePassed TypeOfRecord SERPID QueryID ListOfTerms ListOfURLsAndDomains
    out.writerow(["SessionId", "QueryTime", "Type", "SerpId", "QueryId", "Terms", "URLS"])
    for line in content:
        fields = line.strip().split("\t")
        sessionId = fields[0]
        second = fields[1]
        third = fields[2]
                   
        if third == "Q" or third == "T":
            urls = fields[6:]
            out.writerow([sessionId, second, third, fields[3], fields[4], fields[5], urls])    
        else:
            continue
    fout.close()

def stable3(file, outname="stable3.csv"):
    content = gzip.open(file, "rb")
    fout = open(outname, "wb")
    out = csv.writer(fout)

    #SessionID TimePassed TypeOfRecord SERPID URLID
    out.writerow(["SessionId", "ClickTime", "Type", "SerpId", "UrlId"])
    for line in content:
        fields = line.strip().split("\t")
        sessionId = fields[0]
        timePassed = fields[1]
        rtype = fields[2]
                        
        if rtype == "C":
            serpId = fields[3]
            urlId = fields[4]
            out.writerow([sessionId, timePassed, rtype, serpId, urlId])    
        else:
            continue
    fout.close()

#stable1(file=inputfile)
#stable2(file=inputfile)
#stable3(file=inputfile)

data1 = pd.read_csv(filetouse % (1))
data2 = pd.read_csv(filetouse % (2))
data3 = pd.read_csv(filetouse % (3))

data = pd.merge( data1, pd.merge(data2, data3, on=["SessionId", "SerpId"], how="outer"), on =["SessionId"])
del data["Type"]
#del data["Type_x"]
del data["Type_y"]

clicksQPU = data.groupby(["UserId", "QueryId", "UrlId"]).size()
clicksQU = data.groupby(["UserId", "QueryId"]).size()

gx = data.groupby("QueryId")["UrlId"].apply(lambda x: list(x))
clicksQP = data.groupby(["QueryId", "UrlId"]).size()
clicksQ = data.groupby(["QueryId"]).size()

def clickEntropy(queryId):
    urls = gx.get(queryId)
    if not urls:
        return 0.0

    query = clicksQ.get((queryId))
    if not query:
        return 0.0

    entropy = 0
    for u in urls:
        queryUrl = clicksQP.get((queryId, u))
        if queryUrl:
            pOfpq = 1.0*queryUrl / query
            entropy += (-pOfpq * math.log(pOfpq,2))

    return entropy

def printit(sessionId, entropy, reranked, urls, f=sys.stdout, entropyThresold=1.0):
    if entropy > entropyThresold:
        for url in reranked:
            print("%d,%s" % (sessionId, url), file=f)
    else:
        for url in urls:
            print("%d,%s" % (sessionId, url), file=f)

def s(l):
    r = []
    for li in l:
        r.append(li.split(",")[0])
    return np.array(r)

#returns the domain
def s1(l):
    r = []
    for li in l:
        r.append(li.split(",")[1])
    return np.array(r)

def getDomain(urlid, urls, domains):
    if np.isnan(urlid):
        return np.NAN
    urlid = str(int(round(urlid)))
    for u, d in zip(urls, domains):
        if u == urlid:
            return d
    return np.NAN

data.head().apply( lambda row: getDomain(row["UrlId"], s(ast.literal_eval(row["URLS"])), s1(ast.literal_eval(row["URLS"]))), axis=1)

#4 hours to process
data["domainId"] = data.apply( lambda row: getDomain(row["UrlId"], s(ast.literal_eval(row["URLS"])), s1(ast.literal_eval(row["URLS"]))), axis=1)

dT = data[data["Type_x"] == "T"]
clicksQPD = data.groupby(["UserId", "QueryId", "domainId"]).size()

def spclick(user, query, url, domain=False, beta=0.5):
    userQuery = clicksQU.get((user, query))
    if domain:
        userQueryUrl =  clicksQPD.get((user, query, url)) 
    else:
        userQueryUrl =  clicksQPU.get((user, query, url)) 
    if userQueryUrl:
        return userQueryUrl / (userQuery + beta)
    else:
        return 0

def rerankit(qid, uid, urls, domain=False, b=0.5):
    #Try to re-rank
    urlscores = []
    for urlid in urls:
        if domain:
            urlscores.append( spclick(uid, qid, int(urlid), domain=True, beta=b) )
        else:
            urlscores.append( spclick(uid, qid, int(urlid), domain=False, beta=b) )

    urls = np.array(urls)
    urlscores = np.array(urlscores)
    
    #sort urls without changing the original order
    nonzeros = urlscores[urlscores != 0].shape[0]
    if nonzeros > 0:
        indexes = np.concatenate(( np.argsort(urlscores)[::-1][0:nonzeros], np.argsort(urlscores)[0:-nonzeros] ))
    else:
        indexes = np.arange(0,10)
    #returns the ranked list of urls
    return str(list(urls[indexes]))


dT["dreranked"] = dT.apply( lambda row: rerankit(row["domainId"], row["UserId"], s1(ast.literal_eval(row["URLS"])), domain=True ), axis=1)

dT["entropy"] = dT.apply( lambda row: clickEntropy(row["QueryId"]), axis=1)
dT["reranked"] = dT.apply( lambda row: rerankit(row["QueryId"], row["UserId"], s(ast.literal_eval(row["URLS"]))), axis=1)

f1 = open("out", "w")
print >>f1 ,"SessionID,URLID"
dT.apply( lambda row: printit( row["SessionId"], row["entropy"], ast.literal_eval(row["reranked"]), s(ast.literal_eval(row["URLS"])), f1, entropyThresold=1.0) , axis=1)
f1.close()


