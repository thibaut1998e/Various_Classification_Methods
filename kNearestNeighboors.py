import numpy as np
import loadData


kNN, softKnn = "kNN", "softKnn"

def kNearestNeighboors(trainingSet, k, targetVector, listClass, targetClass):
    def distanceToTargetVector(labelledVector):
        return dist(labelledVector.vector, targetVector)
    trainingSet.sort(key=distanceToTargetVector)
    numberOfNeighboorsPerClass = dict(zip(listClass, [0]*len(listClass)))

    for i in range(k):
        numberOfNeighboorsPerClass[trainingSet[i].getValue(targetClass)] += 1

    bestClass = ""
    bestNbNeig = 0
    for classe in listClass:
        if numberOfNeighboorsPerClass[classe] > bestNbNeig:
            bestNbNeig = numberOfNeighboorsPerClass[classe]
            bestClass = classe
    return bestClass

def succeedPercentageKNN(trainingSet, validationSet, listClass, targetClass = "", k=20, sigma = 0.3, algorithm = kNN):
    confMat = np.zeros((len(listClass),len(listClass)))
    if targetClass == "":
        confMat = np.zeros((len(listClass),len(listClass)))
    nbSuccess = 0
    i = 0
    for x in validationSet:
        #print(i)
        i += 1
        if algorithm == kNN:
            predictedClass = kNearestNeighboors(trainingSet, k, x.vector, listClass, targetClass)
        if algorithm == softKnn:
            predictedClass = kNearestNeighboorsVariant(trainingSet, sigma, x.vector, listClass, targetClass)
        confMat[x.getValue(targetClass)][predictedClass] += 1
        accuracy = sum([confMat[i][i] for i in range(len(confMat))]) / len(validationSet)
    return accuracy, confMat

def findBestValueOfK(trainingSet, validationSet, listClass, kmin, kmax, targetClass=""):
    bestSucceedPercentage = 0
    bestK = 0
    listSucceedPercentages = []
    listK = []
    for k in range(kmin, kmax):
        succeedPercentage = succeedPercentageKNN(trainingSet, validationSet, listClass, k=k, targetClass=targetClass)
        print(k)
        print(succeedPercentage)
        if succeedPercentage > bestSucceedPercentage:
            bestSucceedPercentage = succeedPercentage
            bestK = k
        listK.append(k)
        listSucceedPercentages.append(succeedPercentage)
    return bestK, listK, listSucceedPercentages

def dist(v1, v2):
    return np.sqrt(sum([(v1[i]-v2[i])**2 for i in range(len(v1))]))


def kNearestNeighboorsVariant(trainingSet, sigma, targetVector, listClass, targetClass):
    scoresOfClass = dict(zip(listClass, [0]*len(listClass)))
    for x in trainingSet:
        scoresOfClass[x.getValue(targetClass)] += np.exp(-dist(targetVector,x.vector)**2/sigma**2)
    bestClass = ""
    bestScore = 0
    for classe in listClass:
        if scoresOfClass[classe] > bestScore:
            bestScore = scoresOfClass[classe]
            bestClass = classe
    return bestClass

def findBestValueOfSigma(trainingSet, validationSet, listClass, sigmaMin, sigmaMax, stepSigma, targetClass=""):
    bestSucceedPercentage = 0
    bestSigma = 0
    listSucceedPercentages = []
    listSigma = []
    for sigma in np.arange(sigmaMin, sigmaMax, stepSigma):
        succeedPercentage = succeedPercentageKNN(trainingSet, validationSet, listClass,
                                                 sigma=sigma, algorithm=softKnn, targetClass=targetClass)
        print(sigma)
        print(succeedPercentage)
        if succeedPercentage > bestSucceedPercentage:
            bestSucceedPercentage = succeedPercentage
            bestSigma = sigma
        listSigma.append(sigma)
        listSucceedPercentages.append(succeedPercentage)
    return bestSigma, listSigma, listSucceedPercentages




