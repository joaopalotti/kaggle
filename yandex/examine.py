
from __future__ import division
import gzip, sys

content = gzip.open("smalldata.gz", "rb").readlines()

#print content

for line in content:
    
    field = line.strip().split("\t")
    sessionId = field[0]
    secondField = field[1]
    urls = []
    urlDomain = {}

    if secondField == "M":
        typeOfRecord = "M"
        day = field[2] 
        userid = field[3]

    else:
        timePassed = field[1]
        typeOfRecord = field[2]

        if typeOfRecord == "Q" or typeOfRecord == "T":
            serpid = field[3]
            queryid = field[4]
            terms = field[5]
            urls = field[6:]
            urldomain = [u.split(",") for u in urls]
            for (url, domain) in urldomain:
                if url in urlDomain and urlDomain[url] != domain:
                    print "achei uma safada aqui:", url, domain
                else:
                    urlDomain[url] = domain
        
        elif typeOfRecord == "C":
            serip = field[3]
            urlid = field[4]

        else:
            print "Ops...something wrong here:", line
            sys.exit(0)

    print urls

# check if for each document only one domain is found



### Baseline



