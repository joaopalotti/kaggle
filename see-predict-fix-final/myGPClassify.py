#!/usr/bin/env python
# encoding: utf-8

from __future__ import division
import sys, random, operator, math, csv
from deap import algorithms, base, creator, tools, gp
from optparse import OptionParser
from sklearn.cross_validation import  KFold, train_test_split
import pickle
import numpy as np

usingScoop = True
training =  True

'''
The goal of this version is to separate the Simple English from the English Wikipedia as much as possible.
'''

def kernelCalc(func, t):          
    return func(t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7],t[8],t[9],t[10],t[11],t[12],t[13],t[14],t[15],t[16],t[17],t[18])

def staticLimitCrossover(ind1, ind2, heightLimit, toolbox): 
    # Store a backup of the original individuals 
    keepInd1, keepInd2 = toolbox.clone(ind1), toolbox.clone(ind2) 

    # Mate the two individuals 
    # The crossover is done in place (see the documentation) 
    gp.cxOnePoint(ind1, ind2)

    # If a child is higher than the maximum allowed, then 
    # it is replaced by one of its parent 
    if ind1.height > heightLimit: 
        ind1[:] = keepInd1 
    if ind2.height > heightLimit: 
        ind2[:] = keepInd2

    return ind1, ind2

def staticLimitMutation(individual, expr, heightLimit, toolbox): 
    # Store a backup of the original individual 
    keepInd = toolbox.clone(individual) 

    # Mutate the individual 
    # The mutation is done in place (see the documentation) 
    gp.mutUniform(individual, expr) 

    # If the mutation sets the individual higher than the maximum allowed, 
    # replaced it by the original individual 
    if individual.height > heightLimit: 
        individual[:] = keepInd  

    return individual,

def safeDiv(left, right):
    
    if right == 0.0:
        return 0.0
    else:
        return abs(left / right)

def safeDiff(a,b):
    if a - b > 0:
        return a - b
    else:
        return 0.0

def absDiff(a,b):
    return abs(a-b)

def safeExp(a):
    v = math.exp(a)
    if math.isnan(v) or math.isinf(v):
        return a
    else:
        return v

def safeLog(a):
    if abs(a) >= 1:
        return math.log(abs(a))
    else:
        return 0.0

def sin(a):
    return math.sin(a)

def cos(a):
    return math.cos(a)

def getInputFile(fileName):
    
    file = open(fileName,"rb")
    reader = csv.reader(file, delimiter=',', quotechar ='"', escapechar='\\', doublequote=False)
    
    featureList = [ [ float(element.strip()) for element in line ] for line in reader ]
    
    print "len list = ", len(featureList)
    return featureList

## Create the fitness and individual classes
# The second argument is the number of arguments used in the function
pset = gp.PrimitiveSet("MAIN", 19)
'''
    Arg0  => numWords
    Arg1  => numSentences
    Arg2  => numSyllables
    Arg3  => numberOfPolysyllableWord
    Arg4  => numberOfChars
'''
pset.addPrimitive(safeDiv, 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(absDiff, 2)
pset.addPrimitive(safeDiff, 2)
pset.addPrimitive(safeLog, 1)
pset.addPrimitive(safeExp, 1)
pset.addPrimitive(sin, 1)
pset.addPrimitive(cos, 1)

def myEphemeral():
    return random.random()

pset.addEphemeralConstant(myEphemeral)
#    pset.addTerminal(1)
#    pset.addTerminal(0)

creator.create("Fitness", base.Fitness, weights=(-1.0,))
#creator.create("Fitness", base.Fitness, weights=(-1.0,-1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness, pset=pset)

toolbox = base.Toolbox()
toolbox.register("lambdify", gp.lambdify, pset=pset)

if usingScoop:
    from scoop import futures
    toolbox.register("map", futures.map)

def calculateRMSLE(actuals, predictions):
    predictions[ predictions < 0 ] = 0.0
    return math.sqrt( 1.0 / (3 * predictions.shape[0]) * np.sum(np.power(np.log((predictions + 1)) - np.log((actuals + 1)), 2)))

def calculateRMSLE2(actuals, predictions):
    predictions = np.exp(predictions) - 1
    actuals = np.exp(actuals) - 1
    predictions[ predictions < 0 ] = 0.0
    return math.sqrt( 1.0 / (3 * predictions.shape[0]) * np.sum(np.power(np.log((predictions + 1)) - np.log((actuals + 1)), 2)))

def calculateRMSLE3(actuals, predictions):
    predictions = np.power(2.0, predictions) - 1
    actuals = np.power(2.0, actuals) - 1
    predictions[ predictions < 0 ] = 0.0
    return math.sqrt( 1.0 / (3 * predictions.shape[0]) * np.sum(np.power(np.log((predictions + 1)) - np.log((actuals + 1)), 2)))

def final_test(individual):
    func = toolbox.lambdify(expr=individual)
    funcResult = []
    for t in X_test:
        funcResult.append(kernelCalc(func, t))
    
    pred = np.array(funcResult)
    pred = np.exp(pred) - 1
    pred[ pred < 0 ] = 0.0
    return pred,

def test_evaluate(individual):
    func = toolbox.lambdify(expr=individual)
    funcResult = []
    for t in X_test:
        funcResult.append(kernelCalc(func, t))
    
    fitness = calculateRMSLE2(y_test, np.array(funcResult))
    return fitness,

def evaluate(individual):
    func = toolbox.lambdify(expr=individual)
    funcResult = []
    alpha = 0.001
    regularization = 0.0
    correct, total = 0.0, 0
    #print "Function --> ", func
    
    for t in X_train:
        funcResult.append(kernelCalc(func, t))
    
    #fitness = calculateRMSLE(y_train, np.array(funcResult))
    fitness = calculateRMSLE2(y_train, np.array(funcResult))
    #fitness3 = calculateRMSLE3(transf_views, fr)


    #if math.isinf(fitness):
    #    print "Infinit"
    #    import ipdb
    #    ipdb.set_trace()

    #if math.isnan(fitness):
    #    print "NAN"
    #    import ipdb
    #    ipdb.set_trace()

    return fitness, #len(individual)


def main(ngen, npop, mutpb, cxpb, seedValue, tournSize, heightMaxCreation, heightMexNew, heightLimit):
    toolbox.register("evaluate", evaluate)
    toolbox.register("test_evaluate", test_evaluate)
    toolbox.register("final_test", final_test)
    toolbox.register("select", tools.selTournament, tournsize=tournSize)
    toolbox.register("mate", staticLimitCrossover, heightLimit=heightLimit, toolbox=toolbox)
    toolbox.register("expr_mut", gp.genGrow, min_=0, max_=heightMaxNew)
    toolbox.register("mutate", staticLimitMutation, expr=toolbox.expr_mut, heightLimit=heightLimit, toolbox=toolbox)
    toolbox.register("expr", gp.genRamped, pset=pset, min_=0, max_=heightMaxNew)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
 
    #here starts the algorithm
    random.seed(seedValue)
    pop = toolbox.population(n=npop)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", tools.mean)
    stats.register("std", tools.std)
    stats.register("min", min)
    stats.register("max", max)

    #kf = KFold( X.shape[0], n_folds=10, random_state=0)
    #fitnessTest = []
    #for train_index, test_index in kf:

        #X_train, X_test = X[train_index], X[test_index]
        #y_train, y_test = logy_views[train_index], logy_views[test_index]

    algorithms.eaSimple(pop, toolbox, cxpb, mutpb, ngen, stats, halloffame=hof)

    print stats, hof
    print "Fitness in training = ", map(toolbox.evaluate, hof)

    if training:    
        fitnessInTest = map(toolbox.test_evaluate, hof)
        print "Fitness in test = %.4f" % ( fitnessInTest[0][0])

    #    import ipdb
    #    ipdb.set_trace()
    else:
        values  = np.array(map(toolbox.final_test, hof)[0][0])
    
        outfile = open("gp.csv", "wb")
        open_file_object = csv.writer(outfile)
        open_file_object.writerow(["id","num_views","num_votes","num_comments"])
        open_file_object.writerows(zip(np.array(ids), values, values, values))
        outfile.close()

    #algorithms.eaMuPlusLambda(pop, toolbox, npop, npop + 50, cxpb, mutpb, ngen, stats, halloffame=hof)

    #print pop, stats, hof
    #return fitnessInTest[0][0]
    
(X, Xtest, y_views, y_comments, y_votes, ids, times, hours, Xcity, Xsource, Xtags, h, months, lati, longi, test_times, test_Xcity, test_Xsource, test_Xtags, test_hours, test_h, test_months, test_lati, test_longi) = pickle.load( open( "save.p", "rb" ) )
logy_views =  np.log(y_views + 1.0)
transf_views = np.log2(y_views + 1.0)

from sklearn.preprocessing import scale

if training:
    X_train, X_test, y_train, y_test = train_test_split(scale(X), logy_views, test_size=0.40, random_state=42)
else:
    X_train = scale(np.column_stack((times, Xcity, Xsource, h, months, lati, longi, Xtags)))
    X_test = scale(np.column_stack((test_times, test_Xcity, test_Xsource, test_h, test_months, test_lati, test_longi, test_Xtags)) )
    y_train = logy_views


if __name__ == "__main__":

    op = OptionParser(version="%prog 0.001")
    #op.add_option("--simple", "-s", action="store", type="string", dest="simpleFileName", help="File Name for the Simple English Wikipedia Dataset.", metavar="FILE")
    #op.add_option("--en", "-e", action="store", type="string", dest="enFileName", help="File Name for the English Wikipedia Dataset.", metavar="FILE")
    op.add_option("--gen", "-g", action="store", type="int", dest="ngen", help="Number of generations.", metavar="GEN", default=50)
    op.add_option("--pop", "-p", action="store", type="int", dest="npop", help="Number of individuals.", metavar="POP", default=100)
    op.add_option("--mutb", "-m", action="store", type="float", dest="mutpb", help="Probability of multation.", metavar="PROB", default=0.10)
    op.add_option("--cxpb", "-c", action="store", type="float", dest="cxpb", help="Probability of crossover.", metavar="PROB", default=0.90)
    op.add_option("--seed", "-s", action="store", type="int", dest="seed", help="Random Seed.", metavar="SEED", default=29)
    op.add_option("--tsize", "-t", action="store", type="int", dest="tsize", help="Tournament Size.", metavar="TSIZE", default=2)
    
    op.add_option("--hmc", action="store", type="int", dest="hcreation", help="Height for creation.", metavar="HEIGHT", default=5)
    op.add_option("--hnew", "-n", action="store", type="int", dest="hnew", help="Height max for creation.", metavar="HEIGHT", default=1)
    op.add_option("--hlim", "-l", action="store", type="int", dest="hlim", help="Height limit.", metavar="HEIGHT", default=30)
    (opts, args) = op.parse_args()
    
    heightMaxCreation = 5
    heightMaxNew = 1
    heightLimit = 30
    
    if len(args) > 0:
        op.error("this script takes no arguments.")
        sys.exit(1)

    main(opts.ngen, opts.npop, opts.mutpb, opts.cxpb, opts.seed, opts.tsize, opts.hcreation, opts.hnew, opts.hlim)

