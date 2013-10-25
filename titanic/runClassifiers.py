from __future__ import division

import pickle, sys
import random
import numpy as np
from optparse import OptionParser
from collections import defaultdict
import logging

#classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier

#My classes
from classifiers import classify, plotGraph, parallelClassify, getCurves
from auxClassifier import preprocessing, shuffleIndices, vectorizeData, calculateBaselines #, getSubLists
from createFeatureVector import Example

### HOW TO USE:
# python runClassifiers.py -h
# python runClassifiers.pt --preprocessing=[normalize|scale|minmax|nothing] -b [forceBalance|-1] -g [proportional|-1] -s [nseed]"

#TODO: create a parameter for this flag
useIntegral = True

CSVM = 10
SVMMaxIter=10000
SVMWeight = "auto" # [default: None] 
SVMGamma = 0
SVMKernel= "linear" 
#SVMKernel= "rbf"

etcEstimators = 120
ROCNAME="ROC"
PRECRECALLNAME="PrecAndRecall"

classifyParameters = {"KNN-K": 20, "ETC-n_estimators": etcEstimators, "SVM-cacheSize": 2000, "SVM-kernel": SVMKernel, "SVM-C": CSVM, "SVM-maxIter":SVMMaxIter, "SVM-gamma":SVMGamma, "LR-C":1000, "ETC-criterion": "entropy", "ETC-max_features":None, "DT-criterion": "entropy", "DT-max_features":None, "SVM-class_weight":SVMWeight} 

gridETC = [{'criterion': ['entropy'], 'max_features': [None], "n_estimators":[10,100,1000,10000]}]
gridKNN = [{'n_neighbors': [1,5,10,15,20,50,100], 'algorithm': ["auto"]}]
gridLR = [{'C': [1,1000,10000,10000000], 'penalty': ["l1", "l2"]}]
gridDT = [{'criterion': ["gini","entropy"], 'max_features': ["auto", None, "log2"]}]
gridSVM = [{'kernel': ['rbf'], 'gamma': [1, 0, 1e-1, 1e-2, 1e-3, 1e-4], 'C': [0.1,1,10,1000000]},\
            {'kernel': ['linear'], 'C': [0.01, 0.1,1,10,1000000]}]
gridSGD = [{}] #http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier

def voting(listProbas):

    examplePos, exampleNeg = [0] * 418, [0] * 418
    for subList in listProbas:
        for (i,e) in zip(range(len(subList)), subList):
            exampleNeg[i] += e[0]
            examplePos[i] += e[1]
    mergedYTest = []
    for neg, pos in zip(exampleNeg, examplePos):
        if neg > pos:
            mergedYTest.append(0)
        else:
            mergedYTest.append(1)

    return mergedYTest 

def writeOutput(file, results):
    with open(file, "w") as f:
        f.write("PassengerId,Survived\n")
        for (n, r) in zip(range(892,892+len(results)), results):
            f.write("%d,%d\n" % (n, r))
            #f.write("%d,%d,%s\n" % (n, r, t[0]))

def transformeInDict(exampleVector, nseed, n=-1, proportional=-1):
    listOfDicts = list()
    listOfLabels = list()

    p = shuffleIndices(len(exampleVector), nseed)
    if proportional > 0:
        n = int( int(proportional)/100.0 * len(exampleVector) )

    for (v, (_,example)) in zip(p, exampleVector):
        if n >= 0 and v >= n:
            continue 
        listOfDicts.append(example.toDict())
        listOfLabels.append(example.label)
        #print udict  #### Check how this features are related with the features calculated by the random tree method
    return listOfDicts, listOfLabels

def runClassify(preProcessingMethod, forceBalance, proportional, nseed, explanation, gridSearch, generatePickle, hasPlotLibs, paralled, nJobs, listOfClassifiers, outfileName, nCV, measureProbas):
   
    positiveOutputFile = "positive-%s.pk" % (explanation)
    validationPosOutputFile = "positive-validation.pk"
    negativeOutputFile = "negative-%s.pk" % (explanation)
    validationNegOutputFile = "negative-validation.pk"
    testOutputFile = "test-%s.pk" % (explanation)

    logging.info("Using seed: %d", nseed)
    logging.info("Loading: %s and %s", positiveOutputFile, negativeOutputFile)
    logging.info("Processing method used: %s", preProcessingMethod)

    if forceBalance > 0:
        logging.warning("Forcing only %s examples for each dataset",forceBalance)

    if proportional > 0:
        logging.warning("Using proportional representation. %s percente of the base.",proportional)
    
    if forceBalance > 0 and proportional > 0:
        logging.error("ERROR! YOU SHOULD CHOOSE OR FORCEBALANCE OR PROPORTIONAL DATA!")
        print "ERROR! YOU SHOULD CHOOSE OR FORCEBALANCE OR PROPORTIONAL DATA!"
        exit(0)

    ####
    ### Load Datasets
    ##
    #
    logging.info("Loading the datasets...")
    with open(negativeOutputFile, 'rb') as input:
        negativeFV = pickle.load(input)
    
    with open(validationNegOutputFile, 'rb') as input:
        validationNegFV = pickle.load(input)
    
    with open(positiveOutputFile, 'rb') as input:
        positiveFV = pickle.load(input)
    
    with open(validationPosOutputFile, 'rb') as input:
        validationPosFV = pickle.load(input)

    with open(testOutputFile, 'rb') as input:
        testFV = pickle.load(input)
    logging.info("Loaded")

    testFV = sorted(testFV.iteritems(), key=lambda k: int(k[0])) 

    logging.info("Transforming datasets into Dictionaries...")
    ld1, ll1 = transformeInDict(sorted(negativeFV.iteritems()), nseed, forceBalance, proportional)
    ld2, ll2 = transformeInDict(sorted(positiveFV.iteritems()), nseed, forceBalance, proportional)
    ldTest, llTest = transformeInDict(testFV, nseed, forceBalance, proportional)

    valldNeg, valllNeg = transformeInDict(sorted(validationNegFV.iteritems()), nseed, forceBalance, proportional)
    valldPos, valllPos = transformeInDict(sorted(validationPosFV.iteritems()), nseed, forceBalance, proportional)
    valY = np.array( valllNeg + valllPos)
    valDicts = valldNeg + valldPos
    
    logging.info("Transformed")
    
    listOfDicts = ld1 + ld2
    listOfLabels = ll1 + ll2
    y = np.array( listOfLabels )
    
    greatestClass = 0 if len(ll1) > len(ll2) else 1
    y_greatest =  np.array((len(ll1) + len(ll2)) * [greatestClass] )

    logging.info("Using %d positive examples -- class %s" % (len(ll1), ll1[0]))
    logging.info("Using %d negative examples -- class %s" % (len(ll2), ll2[0]))
    
    baselines = calculateBaselines(y, y_greatest)
    
    logging.info("Vectorizing dictionaries...")
    vec, X_noProcess = vectorizeData(listOfDicts) 
    if X_noProcess != []:
        logging.info("Feature Names: %s", vec.get_feature_names())
    logging.info("Vectorized")
   
    logging.info("Preprocessing data")
    X = preprocessing(X_noProcess, preProcessingMethod)
    #print "X_noProcess ----> ", X_noProcess
    #print "X ---> ", X
    logging.info("Data preprocessed")

    #Prepare Test data: 
    Xtest = vec.transform(ldTest).toarray()
    Xtest = preprocessing(Xtest, preProcessingMethod)

    valX = vec.transform(valDicts).toarray()
    valX = preprocessing(valX, preProcessingMethod)
    
    ####
    ### Shuffer samples  (TODO: Cross-validation)
    ##
    #
    logging.info("Shuffling the data...")
    n_samples = len(y)
    newIndices = shuffleIndices(n_samples, nseed)
    X = X[newIndices]
    y = y[newIndices]

    n_samples_val = len(valY)
    newIndices = shuffleIndices(n_samples_val, nseed)
    valX = valX[newIndices]
    valY = valY[newIndices]

    logging.debug("X - %s", X)
    # Shuffle samples
    logging.info("Shuffled")
    
    ####
    ### Run classifiers
    ##
    #
    precRecall, roc = {}, {}
    results = []

    logging.info("Running classifiers...")
    
    if "dmfc" in listOfClassifiers:
        dmfc = DummyClassifier(strategy='most_frequent')
        results.append(classify(dmfc, "DummyMostFrequent", X, y, nCV, nJobs, baselines, {"measureProbas":measureProbas}, Xtest))
    # ================================================================
    if "nbc" in listOfClassifiers or "nb" in listOfClassifiers:
        nbc = GaussianNB()
        results.append(classify(nbc, "Naive Bayes", X, y, nCV, nJobs, baselines, {"measureProbas":measureProbas}, Xtest))
    # ================================================================
    if "knnc" in listOfClassifiers or "knn" in listOfClassifiers:
        knnc = KNeighborsClassifier(n_neighbors=classifyParameters["KNN-K"])
        results.append(classify(knnc, "KNN", X, y, nCV, nJobs, baselines, {"useGridSearch":gridSearch, "gridParameters":gridKNN, "measureProbas":measureProbas}, Xtest))
    # ================================================================
    if "lrc" in listOfClassifiers or "lgr" in listOfClassifiers or "lr" in listOfClassifiers:
        lrc = LogisticRegression(C=classifyParameters["LR-C"])
        results.append(classify(lrc, "Logistic Regression", X, y, nCV, nJobs, baselines, {"useGridSearch":gridSearch, "gridParameters":gridLR, "measureProbas":measureProbas}, Xtest, valX, valY))
    # ================================================================
    if "dtc" in listOfClassifiers:
        dtc = DecisionTreeClassifier( criterion=classifyParameters["DT-criterion"], max_features=classifyParameters["DT-max_features"] )
        results.append(classify(dtc, "Decision Tree", X, y, nCV, nJobs, baselines, {"useGridSearch":gridSearch, "gridParameters":gridDT, "measureProbas":measureProbas}, Xtest))
    # ================================================================
    if "svmc" in listOfClassifiers or "svm" in listOfClassifiers:
        #if SVMKernel == "linear":
        #    svmc = LinearSVC(C=classifyParameters["SVM-C"], class_weight=classifyParameters["SVM-class_weight"])
        #else:
        #    svmc = SVC(kernel=classifyParameters["SVM-kernel"], cache_size=classifyParameters["SVM-cacheSize"], C=classifyParameters["SVM-C"], max_iter=classifyParameters["SVM-maxIter"], probability=measureProbas, gamma=classifyParameters["SVM-gamma"], class_weight=classifyParameters["SVM-class_weight"])
        #results.append(classify(svmc, "SVM", X, y, nCV, nJobs, baselines, {"useGridSearch":gridSearch, "gridParameters":gridSVM, "measureProbas":measureProbas}, Xtest))
        pass
    # ================================================================
    if "etc" in listOfClassifiers:
        etc = ExtraTreesClassifier(random_state=0, n_jobs=nJobs, n_estimators=classifyParameters["ETC-n_estimators"], criterion=classifyParameters["ETC-criterion"], max_features=classifyParameters["ETC-max_features"])
        results.append(classify(etc, "Random Forest", X, y, nCV, nJobs, baselines, {"tryToMeasureFeatureImportance":measureProbas, "featuresOutFilename":(outfileName + ".pk"), "featureNames":vec.get_feature_names(), "useGridSearch":gridSearch, "gridParameters":gridETC, "measureProbas":measureProbas}, Xtest, valX, valY))
    
    # ================================================================
    if "sgd" in listOfClassifiers:
        sgd = SGDClassifier(n_jobs=nJobs)
        results.append(classify(sgd, "SGD", X, y, nCV, nJobs, baselines, {"featuresOutFilename":(outfileName + ".pk"), "featureNames":vec.get_feature_names(), "useGridSearch":gridSearch, "gridParameters":gridSGD, "measureProbas":measureProbas}, Xtest, valX, valY))

    # ================================================================
    if "gbc" in listOfClassifiers:
        gbc = GradientBoostingClassifier(n_estimators=300,subsample=0.6,max_depth=4,random_state=nseed)
        results.append(classify(gbc, "GBC", X, y, nCV, nJobs, baselines, {"featuresOutFilename":(outfileName + ".pk"), "featureNames":vec.get_feature_names(), "useGridSearch":gridSearch, "gridParameters":gridSGD, "measureProbas":measureProbas}, Xtest, valX, valY))
    # ================================================================
    
    
    precRecall, roc = getCurves(results)
    roc["Random Classifier"] = ([0,1],[0,1])

    plotGraph(precRecall, fileName=PRECRECALLNAME, xlabel="Recall", ylabel="Precision", generatePickle=generatePickle, hasPlotLibs=hasPlotLibs)
    plotGraph(roc, fileName=ROCNAME, xlabel="False Positive Rate", ylabel="True Positive Rate", generatePickle=generatePickle, hasPlotLibs=hasPlotLibs)
   
    fo = open(outfileName, "a")

    listProbas = []
    for r in results:
        clfName = r[0]
        resultMetrics = r[1]
        fo.write("%s, %.3f, %.3f, %.3f, %.3f\n" % (clfName, 100.0*resultMetrics.acc, 100.0*resultMetrics.sf1, 100.0*resultMetrics.mf1, 100.0*resultMetrics.wf1))
        print "%s, %.3f, %.3f, %.3f, %.3f" % (clfName, 100.0*resultMetrics.acc, 100.0*resultMetrics.sf1, 100.0*resultMetrics.mf1, 100.0*resultMetrics.wf1)
        
        yTraining = r[4]
        yTrainingProbas = r[5]
        yTest = r[6]
        yTestProbas = r[7]
        writeOutput(clfName + ".csv", yTest)
        
        listProbas.append(yTestProbas)
        #for t,p in zip(yTest, yTestProbas):
        #    print t, p

    mergedYTest = voting(listProbas)
    writeOutput("merged.csv", mergedYTest)


    fo.close()
    logging.info("Done")

if __name__ == "__main__":
    
    op = OptionParser(version="%prog 2")
    op.add_option("--preprocessing", "-p", action="store", type="string", dest="preProcessing", help="Preprocessing option [normalize|scale|minmax|nothing] --  [default: %default]", metavar="OPT", default="normalize")
    op.add_option("--forceBalance", "-b", action="store", type="int", dest="forceBalance", help="Force balance keeping only X instances of each class.", metavar="X", default=-1)
    op.add_option("--proportional", "-q", action="store", type="int", dest="proportional", help="Force proportion of the data to X%.", metavar="X", default=-1)
    op.add_option("--nseed", "-n", action="store", type="int", dest="nseed", help="Seed used for random processing during classification.  [default: %default]", metavar="X", default=29)
    op.add_option("--explanation", "-e", action="store", type="string", dest="explanation", help="Prefix to include in the created files", metavar="TEXT", default="")
    op.add_option("--gridSearch", "-s", action="store_true", dest="gridSearch", help="Use if you want to use grid search to find the best hyperparameters", default=False)
    op.add_option("--hasPlotLibs", "-c", action="store_true", dest="hasPlotLibs", help="Use if you want to plot Precision Vs Recall and ROC curves", default=False)
    op.add_option("--ignorePickle", "-k", action="store_true", dest="ignorePickle", help="Don't Generate Pickle of plots", default=False)
    op.add_option("--useScoop", "-r", action="store_true", dest="useScoop", help="Use Scoop to run classifier in parallel", default=False)
    op.add_option("--njobs", "-j", action="store", type="int", dest="njobs", help="Number of parallel jobs to run.", metavar="X", default=2)
    op.add_option("--classifiers", "-z", action="store", type="string", dest="classifiers", help="Classifiers to run. Options are dmfc|dsc|duc|nbc|knnc|lrc|dtc|svmc|etc|sgd|gbc", metavar="cl1|cl2|..", default="dmfc|dsc|duc|nbc|knnc|lrc|dtc|svmc|etc|sgd|gbc")
    op.add_option("--logFile", "-l", action="store", type="string", dest="logFile", help="Log filename", default="debug.log")
    op.add_option("--outfileName", "-o", action="store", type="string", dest="outfileName", help="Filename to write the classification output", default="classification.out")
    op.add_option("--nFolds", "-f", action="store", type="int", dest="nFolds", help="Number of folds for the cross-validation process", default=10)
    op.add_option("--measureProbas", "-a", action="store_true", dest="measureProbas", help="Active it if you want to measure probabilities. They are necessary to plot ROC and Precision X Recall curves", default=False)
    
    (opts, args) = op.parse_args()
    if len(args) > 0:
        print "This program does not receive parameters this way: use -h to see the options."

    logger = logging.getLogger('runClassify.py')
    formatter = logging.Formatter('%(asctime)s - %(name)-15s: %(levelname)-8s %(message)s') 
    logging.basicConfig(format='%(asctime)s * %(name)-12s * %(levelname)-8s * %(message)s', datefmt='%m-%d %H:%M', level=logging.DEBUG,\
                        filename=opts.logFile, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    logging.info("Writing DEBUG output in : %s", opts.logFile)
    logging.info("Using Preprocessing: %s", opts.preProcessing)
    logging.info("Forcing Balance: %d", opts.forceBalance)
    logging.info("Proportional: %d", opts.proportional)
    logging.info("Using Grid Search: %d", opts.gridSearch)
    logging.info("NFolds for CV = %s", opts.nFolds)
    logging.info("Has plot libs: %d", opts.hasPlotLibs)
    logging.info("Generating Pickle = %d", not opts.ignorePickle)
    logging.info("Running in parallel = %d", opts.useScoop)
    logging.info("Njobs = %d", opts.njobs)
    logging.info("Classifiers = %s", opts.classifiers)
    listOfClassifiers = opts.classifiers.split("|")

    if "svm" in listOfClassifiers or "svmc" in listOfClassifiers and opts.preProcessing != "scale":
        logging.warning("You are using SVM --- you should consider process the data using the 'scale' preprocessing method")

    #uncomment if it is necessary to see the complete numpy arrays
    #np.set_printoptions(threshold='nan')
    
    runClassify(opts.preProcessing, opts.forceBalance, opts.proportional, opts.nseed, opts.explanation, opts.gridSearch, not opts.ignorePickle, opts.hasPlotLibs, opts.useScoop, opts.njobs, listOfClassifiers, opts.outfileName, opts.nFolds, opts.measureProbas)
    

