from __future__ import division
from itertools import groupby
from collections import Counter, defaultdict 
from optparse import OptionParser
import sys, pickle
import csv as csv
import numpy as np


class Example:
    def __init__(self, label, pclass, age, sex, embarked, sibsp, parch, fare, title, ticket):
        self.label = label
        self.pclass = pclass
        self.age = age
        self.sex = sex
        self.embarked = embarked
        self.sibsp, self.parch = sibsp, parch
        self.fare = fare
        self.title = title
        self.ticket = ticket

    def __rep__(self):
        return ",".join([self.name, self.title, self.sex, self.age, self.pclass])

    def toDict(self):
        featuresToUse = {}
        featuresToUse["Pclass"] = self.pclass
        featuresToUse["Age"] = self.age
        featuresToUse["Sex"] = self.sex
        featuresToUse["Embarked"] = self.embarked
        featuresToUse["Sibsp"] = self.sibsp
        featuresToUse["Parch"] = self.parch
        featuresToUse["Sibsp"] = self.sibsp
        featuresToUse["Fare"] = self.fare
        #featuresToUse["Title"] = self.title
        featuresToUse["Title_miss"] = self.title == 1 
        featuresToUse["Title_mrs"] = self.title == 2 
        featuresToUse["Title_mr"] = self.title == 3
        featuresToUse["Title_other"] = self.title >= 4 
        #featuresToUse["Ticket"] = self.ticket
        
        return featuresToUse

class ProblemType:
    def __init__(self, data):
    #PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

        #TODO: decide what to do with invalid values
        self.id = data[0]
        self.label = int(data[1]) if data[1] != None else None
        self.pclass = int(data[2])
        self.name = data[3]
        self.sex = 1 if data[4] == "male" else 0
        self.age = -1 if data[5] == "" else float(data[5])
        self.sibsp = int(data[6])
        self.parch = int(data[7])
        self.ticket = data[8]
        self.fare = -1 if (data[9] == "") else float(data[9]) 

        self.cabin = data[10] 
        #Embarked changed to 0, 1 or 2
        self.embarked = 0 if data[11] == "C" else 1 if data[11] == "Q" else 2

def createDictOfExamples(data, label):
    exampleDict = dict()
    ids = [example.id for example in data]
        
    dictPClass = calculatePClass(data)
    dictAge = calculateAge(data)
    dictSex = calculateSex(data)
    dictEmbarked = calculateEmbarked(data)
    dictSibsp = calculateSibsp(data)
    dictParch = calculateParch(data)
    dictFare = calculateFare(data)
    dictTitle = calculateName(data)
    dictTicket = calculateTicket(data)
    dictCabin = calculateCabin(data)
    
    for id in ids:
        pclass, age, sex, embarked = dictPClass[id], dictAge[id], dictSex[id], dictEmbarked[id]
        sibsp, parch = dictSibsp[id], dictParch[id]
        fare = dictFare[id]
        title = dictTitle[id]
        ticket = dictTicket[id]
        
        exampleDict[id] = Example(label, pclass, age, sex, embarked, sibsp, parch, fare, title, ticket)

    return exampleDict

#### ========================= METRICS ============================ #####
def calculatePClass(data):
    ids_pclass = [(example.id, example.pclass) for example in data]
    return dict(ids_pclass)

def calculateAge(data):
    ids_age = [(example.id, example.age) for example in data]
    meanAge = np.mean( [d[1] for d in ids_age if d[1] != -1] )
    #print meanAge
    #missing_age = [ (e, a) for (e,a) in ids_age if a == -1]
    #print missing_age
    ids_age = [ (e, meanAge if a == -1 else a) for (e,a) in ids_age]

    return dict(ids_age)

def calculateSex(data):
    ids_sex = [(example.id, example.sex) for example in data]
    return dict(ids_sex)

def calculateEmbarked(data):
    ids_embarked = [(example.id, example.embarked) for example in data]
    return dict(ids_embarked)

def calculateSibsp(data):
    ids_sibsp = [(example.id, example.sibsp) for example in data]
    return dict(ids_sibsp)

def calculateParch(data):
    ids_parch = [(example.id, example.parch) for example in data]
    return dict(ids_parch)

def calculateFare(data):
    ids_fare = [(example.id, example.fare) for example in data]
    meanFare = np.mean( [d[1] for d in ids_fare if d[1] != -1] )
    ids_fare = [ (e, meanFare if a == -1 else a) for (e,a) in ids_fare]

    return dict(ids_fare)

def calculateName(data):
    ids_name = [(example.id, example.name, example.label) for example in data]
    idsTitle = {}
    for id, name, label in ids_name:
        if ", Miss." in name or ", Ms." in name or ", Dona.":
            title = 1
        elif ", Mrs." in name:
            title = 2
        elif ", Mr." in name:
            title = 3
        elif ", Master" in name:
            title = 4
        elif ", Rev" in name:
            title = 5
        elif ", Dr." in name:
            title = 6
        elif ", Major." in name or ", Col." in name:
            title = 6
        # all survived and are not present in the test set
        elif ", Mme." in name or "Lady." in name or "Sir." in name or "Mlle." in name or "the Countess." in name:
            title = 99
        # all died and are not present in the test set
        elif ", Don." in name or "Capt." in name or "Jonkheer." in name:
            title = 100
        else:
           print name, label
        idsTitle[id] = title

    return idsTitle

def calculateTicket(data):
    ids_ticket = [(example.id, example.ticket) for example in data]
    #ids_ticket = [(example.id, example.ticket, example.label) for example in data]
    
    #for id, ticket, label in ids_ticket:
    #    print id, ticket, label

    return dict(ids_ticket)

def calculateCabin(data):
    ids_cabin = [(example.id, example.cabin, example.label) for example in data]
    
    #for id, cabin, label in ids_cabin:
    #    print id, cabin, label

    return dict([])


##### ========Read input file=================================================================== #####

def readInput(filename, labelToFilter=None):
    csvInput = csv.reader(open(filename, 'rb')) 
    header = csvInput.next() 
    if labelToFilter != None:
        data = [row for row in csvInput if int(row[1]) == labelToFilter]
        data = [ProblemType(row) for row in data]
    else:
        data = []
        for row in csvInput:
            row.insert(1, None)
            data.append(row)
        data = [ProblemType(row) for row in data]
        print len(data)

    return data

##### ========================================================================================== #####

def createFV(filename, label):
    data = readInput(filename, label)
    exampleDict = createDictOfExamples(data, label)
    
    return exampleDict

def readData(filename, explanation):
       
    #find file and read it:
    negativeExamples = createFV(filename, 0)
    positiveExamples = createFV(filename, 1)
    testExamples = createFV("test.csv", None)

    negativeOutputFile = "negative-%s.pk" % (explanation)
    positiveOutputFile = "positive-%s.pk" % (explanation)
    testOutputFile = "test-%s.pk" % (explanation)
   
    ####### Save and Load the Features
    with open(negativeOutputFile, 'wb') as output:
        pickle.dump(negativeExamples, output, pickle.HIGHEST_PROTOCOL)
        print "Created file: %s with %d examples" % (negativeOutputFile, len(negativeExamples))
    
    with open(positiveOutputFile, 'wb') as output:
        pickle.dump(positiveExamples, output, pickle.HIGHEST_PROTOCOL)
        print "Created File: %s with %d examples" % (positiveOutputFile, len(positiveExamples))
    
    with open(testOutputFile, 'wb') as output:
        pickle.dump(testExamples, output, pickle.HIGHEST_PROTOCOL)
        print "Created File: %s with %d examples" % (testOutputFile, len(testExamples))


def testing():
    
    pos, neg = {},{}
    neg = createFV("smallTest", 0)
    pos = createFV("smallTest", 1)
    
    positiveOutputFile = "positive-test.pk"
    negativeOutputFile = "negative-test.pk"
    with open(positiveOutputFile, 'wb') as output:
        pickle.dump(pos, output, pickle.HIGHEST_PROTOCOL)
        print "Created file: %s with %d examples" % (positiveOutputFile, len(pos))
    
    with open("negative-5-test.pk", 'wb') as output:
        pickle.dump(neg, output, pickle.HIGHEST_PROTOCOL)
        print "Created file: %s with %d examples" % (negativeOutputFile, len(pos))

if __name__ == "__main__":

    op = OptionParser(version="%prog 0.1")
    
    op.add_option("--explanation", "-e", action="store", type="string", dest="explanation", help="Prefix to include in the created files", metavar="N", default="")
    op.add_option("--testingOnly", "-t", action="store_true", dest="testingOnly", help="Just to test some new feature", default=False)
    op.add_option("--filename", "-i", action="store", type="string", dest="filename", help="Training filename", metavar="FILE", default="train.csv")

    (opts, args) = op.parse_args()
    if len(args) > 0:
        print "This program does not receive parameters this way: use -h to see the options."
    
    if opts.testingOnly:
        testing()
        sys.exit(0)

    readData(opts.filename, opts.explanation)

    
