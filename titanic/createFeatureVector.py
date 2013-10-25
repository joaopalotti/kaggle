from __future__ import division
from itertools import groupby
from collections import Counter, defaultdict 
from optparse import OptionParser
import sys, pickle
import csv as csv
import numpy as np

import random
random.seed(0)

class Example:
    def __init__(self, label, pclass, age, sex, embarked, sibsp, parch, fare, title, ticket, fareBin, first, last, cabinType, cabinNumber):
        self.label = label
        self.pclass = pclass
        self.age = age
        self.sex = sex
        self.embarked = embarked
        self.sibsp, self.parch = sibsp, parch
        self.fare = fare
        self.fareBin = fareBin
        self.title = title
        self.firstName = first
        self.lastName = last
        self.ticket = ticket
        self.cabinType = cabinType
        self.cabinNumber = cabinNumber
        self.percentageCabinDead = -1
        self.alone = False

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
        featuresToUse["FareBin"] = self.fareBin
        featuresToUse["DeathInFamily"] = self.percentageDiedInFamily
        featuresToUse["CabinType"] = self.cabinType
        #featuresToUse["CabinSharingDeath"] = self.percentageCabinDead

        #featuresToUse["Title"] = self.title
        featuresToUse["Title_miss"] = (self.title == 1) 
        featuresToUse["Title_mrs"] = (self.title == 2)
        featuresToUse["Title_mr"] = (self.title == 3)
        featuresToUse["Title_other"] = (self.title >= 4)
        #featuresToUse["Ticket"] = self.ticket
        
        #print featuresToUse
        return featuresToUse

class ProblemType:
    def __init__(self, data):
    #PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

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
        self.embarked = 0 if data[11] == "C" else 5 if data[11] == "Q" else 10 if data[11] == "S" else -1

def createDictOfExamples(data, label):
    exampleDict = dict()
    ids = [example.id for example in data]
        
    dictPClass = calculatePClass(data)
    dictAge = calculateAge(data)
    dictSex = calculateSex(data)
    dictEmbarked = calculateEmbarked(data)
    dictSibsp = calculateSibsp(data)
    dictParch = calculateParch(data)
    dictFare, dictFareBin = calculateFare(data)
    dictTitle, dictFirstName, dictLastName = calculateName(data)
    dictTicket = calculateTicket(data)
    dictCabinType, dictCabinNumber = calculateCabin(data)
    
    for id in ids:
        pclass, age, sex, embarked = dictPClass[id], dictAge[id], dictSex[id], dictEmbarked[id]
        sibsp, parch = dictSibsp[id], dictParch[id]
        fare, fareBin = dictFare[id], dictFareBin[id]
        title, first, last = dictTitle[id], dictFirstName[id], dictLastName[id]
        ticket = dictTicket[id]
        cabinType, cabinNumber = dictCabinType[id], dictCabinNumber[id]
        
        exampleDict[id] = Example(label, pclass, age, sex, embarked, sibsp, parch, fare, title, ticket, fareBin, first, last, cabinType, cabinNumber)

    return exampleDict

#### ========================= METRICS ============================ #####
def calculatePClass(data):
    ids_pclass = [(example.id, example.pclass) for example in data]
    return dict(ids_pclass)

def calculateAge(data):
    ids_age = [(example.id, example.age) for example in data]
    meanAge = np.median( [d[1] for d in ids_age if d[1] != -1] )
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
    meanEmbarked = np.round(np.mean([example.embarked for example in data if example.embarked != -1]))
    ids_embarked = [(example.id, meanEmbarked if example.embarked == -1 else example.embarked) for example in data]
    return dict(ids_embarked)

def calculateSibsp(data):
    ids_sibsp = [(example.id, example.sibsp) for example in data]
    return dict(ids_sibsp)

def calculateParch(data):
    ids_parch = [(example.id, example.parch) for example in data]
    return dict(ids_parch)

def calculateFare(data):
    ids_fare = [(example.id, example.fare) for example in data]
    ids_fare_class = [(example.id, example.fare, example.pclass) for example in data]
    
    meanFare = {}
    for pclass in range(1,4):
        meanFare[pclass] = np.mean( [d[1] for d in ids_fare_class if d[1] != -1 and d[2] == pclass] )
    
    ids_fare = [ (e, meanFare[pclass] if a == -1 else a) for (e,a,pclass) in ids_fare_class]

    #define fare bins:
    ids_fare_label = [(example.id, example.fare, example.label) for example in data]
    fareBin = {}
    for id, fare, label in ids_fare_label:
        #print id, fare, label
        if fare < 10:
            bin = 0
        elif fare >= 10 and fare < 30:
            bin = 1
        elif fare >= 30 and fare < 50:
            bin = 2
        elif fare >= 50 and fare < 100:
            bin = 3
        else:
            bin = 4

        fareBin[id] = bin

    return dict(ids_fare), fareBin

def calculateName(data):
    ids_name = [(example.id, example.name, example.label) for example in data]
    idsLastName = {}
    idsTitle = {}
    idsFirstName = {}

    for id, name, label in ids_name:
        if ", Miss." in name or ", Ms." in name or ", Dona." in name:
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
            title = 7
        # all survived and are not present in the test set
        elif ", Mme." in name or "Lady." in name or "Sir." in name or "Mlle." in name or "the Countess." in name:
            title = -1
        # all died and are not present in the test set
        elif ", Don." in name or "Capt." in name or "Jonkheer." in name:
            title = -2
        else:
           print name, label

        lastName = name.split(",")[0].strip()
        firstName = name.split(",")[1].split(".")[1].strip()
        
        idsFirstName[id] = firstName
        idsLastName[id] = lastName
        idsTitle[id] = title
    return idsTitle, idsFirstName, idsLastName

def calculateTicket(data):
    ids_ticket = [(example.id, example.ticket) for example in data]
    #ids_ticket = [(example.id, example.ticket, example.label) for example in data]
    
    #for id, ticket, label in ids_ticket:
    #    print id, ticket, label

    return dict(ids_ticket)

def mapCabin(cabin):
    cabinType = -1
    if "A" in cabin:
        cabinType = 0
    elif "B" in cabin:
        cabinType = 1
    elif "C" in cabin:
        cabinType = 2
    elif "D" in cabin:
        cabinType = 3
    elif "E" in cabin:
        cabinType = 4
    elif "F" in cabin:
        cabinType = 5
    elif "G" in cabin:
        cabinType = 6
    return cabinType

def calculateCabin(data):
    ids_cabin = [(example.id, example.cabin, example.label, example.pclass) for example in data]
    idsType = {}
    idsNumber = {}
    
    pclassCabin = defaultdict(list)
    for _, cabin, _ , pclass in ids_cabin:
        pclassCabin[pclass].append( mapCabin(cabin) )
    
    for pclass, values in pclassCabin.iteritems():
        pclassCabin[pclass] = Counter(values)
    
    for id, cabin, label, pclass in ids_cabin:
        cabinType = mapCabin(cabin) 
        
        if cabinType == -1:
        # if the cabin is not informed, we atribute the cabin to the most common one used in the class the person belongs
            common = pclassCabin[pclass].most_common(1)[0][0]
            cabinType = common
            #print "Pclass = ", pclass, " common = ", common

        idsType[id] = cabinType
        idsNumber[id] = cabin

    return idsType, idsNumber


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


def createValidation(data):
    validation = []
    #print len(data), len(validation)
    #10% for validation
    just10p = int(len(data) / 10)
    for _ in range(just10p):
        val = random.randint(0, len(data))
        validation.append(data[val])
        del data[val]

    print len(data), len(validation)
    return data, validation

def createFV(filename, label, usesValidation=False):
    data = readInput(filename, label)
    validationDict = {}

    if usesValidation:
        data, validation = createValidation(data)
        validationDict = createDictOfExamples(validation, label)

    exampleDict = createDictOfExamples(data, label)
    
    return exampleDict, validationDict

def posProcessName(neg, pos, test, valNeg, valPos):
    # check the number of relatives that are died
    name_IdStatus = defaultdict(list)
    for (id, example) in neg.iteritems():
        name_IdStatus[example.lastName].append([id, example.label])
    
    for (id, example) in pos.iteritems():
        name_IdStatus[example.lastName].append([id, example.label])
    
    for (id, example) in test.iteritems():
        name_IdStatus[example.lastName].append([id, example.label])
   
    for (id, example) in valNeg.iteritems():
        name_IdStatus[example.lastName].append([id, None])
    
    for (id, example) in valPos.iteritems():
        name_IdStatus[example.lastName].append([id, None])

    percentageOfSurvivers = len(pos) / (len(neg) + len(pos))
    for (name, idStatusList) in name_IdStatus.iteritems():
        
        sumPerId = defaultdict(float)
        totalSum = np.mean( [ percentageOfSurvivers if status == None else status for (id, status) in idStatusList ] )
        for id, status in idStatusList:
            #sumPerId[id] = totalSum - status if status != None else totalSum - percentageOfSurvivers
            sumPerId[id] = totalSum 

        for (id, _) in idStatusList:
            if id in neg:
                neg[id].percentageDiedInFamily = sumPerId[id]
            elif id in pos:
                pos[id].percentageDiedInFamily = sumPerId[id]
            elif id in test:
                test[id].percentageDiedInFamily = sumPerId[id]
            
            elif id in valPos:
                valPos[id].percentageDiedInFamily = sumPerId[id]
            elif id in valNeg:
                valNeg[id].percentageDiedInFamily = sumPerId[id]

    return neg, pos, test, valNeg, valPos

def posProcessCabin(neg, pos, test):
    
    cabin_IdStatus = defaultdict(list)
    for (id, example) in neg.iteritems():
        cabin_IdStatus[example.cabinType].append([id, example.label])
    
    for (id, example) in pos.iteritems():
        cabin_IdStatus[example.cabinType].append([id, example.label])
    
    for (id, example) in test.iteritems():
        cabin_IdStatus[example.cabinType].append([id, example.label])
 
    for (cabin, idStatusList) in cabin_IdStatus.iteritems():
        #if not cabin:
        #    continue
        totalSum = np.mean( [0.5 if status == None else status for (id, status) in idStatusList] )

        for (id, _) in idStatusList:
            if id in neg:
                neg[id].percentageCabinDead = totalSum
            elif id in pos:
                pos[id].percentageCabinDead = totalSum
            elif id in test:
                test[id].percentageCabinDead = totalSum
    
    return neg, pos, test

def posProcess(neg, pos, test, valNeg, valPos):
    neg, pos, test, valNeg, valPos = posProcessName(neg, pos, test, valNeg, valPos)
    #neg, pos, test = posProcessCabin(neg, pos, test)
    #neg, pos, test = posProcessAlone(neg, pos, test)
    return neg, pos, test, valNeg, valPos

def readData(filename, explanation):
    #
    #preprocessFamilies(filename, "test.csv")

    #find file and read it:
    negativeExamples, validationNeg = createFV(filename, 0, True)
    positiveExamples, validationPos = createFV(filename, 1, True)
    testExamples, _ = createFV("test.csv", None)

    negativeExamples, positiveExamples, testExamples, validationNeg, validationPos = posProcess(negativeExamples, positiveExamples, testExamples, validationNeg, validationPos)
    
    negativeOutputFile = "negative-%s.pk" % (explanation)
    valNegativeOutputFile = "negative-validation.pk"
    positiveOutputFile = "positive-%s.pk" % (explanation)
    valPositiveOutputFile = "positive-validation.pk"
    testOutputFile = "test-%s.pk" % (explanation)
   
    ####### Save and Load the Features
    with open(negativeOutputFile, 'wb') as output:
        pickle.dump(negativeExamples, output, pickle.HIGHEST_PROTOCOL)
        print "Created file: %s with %d examples" % (negativeOutputFile, len(negativeExamples))
    
    with open(valNegativeOutputFile, 'wb') as output:
        pickle.dump(validationNeg, output, pickle.HIGHEST_PROTOCOL)
        print "Created file: %s with %d examples" % (valNegativeOutputFile, len(validationNeg))
    
    with open(positiveOutputFile, 'wb') as output:
        pickle.dump(positiveExamples, output, pickle.HIGHEST_PROTOCOL)
        print "Created File: %s with %d examples" % (positiveOutputFile, len(positiveExamples))
    
    with open(valPositiveOutputFile, 'wb') as output:
        pickle.dump(validationPos, output, pickle.HIGHEST_PROTOCOL)
        print "Created file: %s with %d examples" % (valPositiveOutputFile, len(validationPos))
    
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

    
