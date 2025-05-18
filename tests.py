import copy as cp
import matplotlib.pyplot as plt
import numpy as np
import kNearestNeighboors
import neuralNetWork
import loadData
import simpleClassifier
import matplotlib.pyplot as plt


trainingSet, testSet, validationSet = loadData.createSets()
linReg, logReg, perceptron = "LinReg", "LogReg", "Percepron"
kNN, softKnn = "kNN", "softKnn"




def plotSets():
    loadData.plotSet(trainingSet, 1)
    plt.show()
    loadData.plotSet(trainingSet, "")
    plt.show()


def testKnn(algorithm, targetClass, nonLinearFeatures=False):
    trainingSet2, testSet2 = cp.deepcopy(trainingSet), cp.deepcopy(testSet)
    trainingSet2, testSet2 = cp.deepcopy(trainingSet), cp.deepcopy(validationSet)
    if nonLinearFeatures:
        simpleClassifier.addNonLinearFeatures(trainingSet2)
        simpleClassifier.addNonLinearFeatures(testSet2)

    if targetClass == "":
        kNNAllClasses, confMat = kNearestNeighboors.succeedPercentageKNN(trainingSet2, testSet2, [0, 1, 2], k=20, sigma = 0.3, algorithm=algorithm)
        print("the succeed percentage of " + algorithm + " on the test set with is " + str(kNNAllClasses))
        print(confMat)

    else:
        kNN1vsOthers, confMat = kNearestNeighboors.succeedPercentageKNN(trainingSet2, testSet2, [0,1], targetClass = targetClass, k=20, sigma=0.3, algorithm=algorithm)
        print("the succeed percentage of " + algorithm + " of class " +str(targetClass) +  " versus others on the test is " + str(kNN1vsOthers) + " ")
        print(confMat)



def testSimpleClassifier(type, targetClass):
    loadData.plotSet(testSet, targetClass)
    etha = 0.1
    batchSize = 800
    diff = 0.01
    classifier = simpleClassifier.simpleClassifier(2, batchSize, targetClass, etha, diff, type=type)
    classifier.train(trainingSet, validationSet)
    linearClassifierResult = classifier.getSucceedPercentage(testSet)
    printResultSimpleClassifier(linearClassifierResult, type, targetClass, etha, batchSize)
    classifier.plotDecisionBoarder()
    plt.xlabel("feature1")
    plt.ylabel("feature2")
    plt.title("Decision boarder found by linear classifier " + type)
    plt.show()

def testSimpleClassifierWithNonLinearFeatures(type, targetClass):
    batchSize = 800
    etha = 0.01
    diff = 0.01
    trainingSet2, validationSet2, testSet2 = cp.deepcopy(trainingSet), cp.deepcopy(validationSet), cp.deepcopy(testSet)
    simpleClassifier.addNonLinearFeatures(trainingSet2)
    simpleClassifier.addNonLinearFeatures(validationSet2)
    simpleClassifier.addNonLinearFeatures(testSet2)
    classifier = simpleClassifier.simpleClassifier(5, batchSize, targetClass,etha,diff, type=type)
    classifier.train(trainingSet2, validationSet2)
    nonLinearFeaturesResult = classifier.getSucceedPercentage(testSet2)
    printResultSimpleClassifier(nonLinearFeaturesResult, type, targetClass, etha, batchSize, nonLinearFeatures=True)


def printResultSimpleClassifier(succeedPercentage, type, targetClass, etha, batchSize, nonLinearFeatures=False):
    string = type + ", target class : " + str(targetClass) + ", etha=" + str(etha) + ", batch size = " + str(batchSize) \
             + ", succeed percentage = " + str(succeedPercentage)
    if nonLinearFeatures:
        string = "Non linear fetures, " + string
    print(string)


def testOneVersusRest():
    loadData.plotSet(testSet, "")
    batchSize = 800
    etha = 0.1
    oneVersusRestResult = simpleClassifier.oneVersusRest(trainingSet, validationSet, testSet, batchSize, etha)
    #plt.show()
    #oneVersusRestResult = simpleClassifier.oneVersusRest(trainingSet, validationSet, validationSet, batchSize, etha)
    print("One versus Rest, etha = " + str(etha) +", batchSize = " + str(batchSize) + ", succeedPerecentage = " + str(oneVersusRestResult))


def testNeuralNetwork(plot=False):
    neuronesHiddenLayer = 10
    etha = 0.1
    #batchSize = 1 #if you use batch size = 1 set etha = 0.01 (etha is divided by batch size)
    #batchSize = 10
    batchSize = len(trainingSet)
    trainingSet2, validationSet2, testSet2 = cp.deepcopy(trainingSet), cp.deepcopy(validationSet), cp.deepcopy(testSet)
    nbDim = len(trainingSet2[0].vector)
    nbDimOutput = len(trainingSet2[0].labelVector)
    print(nbDimOutput)
    neuralNetWork.normalizeSets(trainingSet2, validationSet2, testSet2)
    neuNet = neuralNetWork.neuralNetWork([nbDim,neuronesHiddenLayer, nbDimOutput], etha,
                                         batchSize=batchSize, e=5, t=0, nbMaxEpochs = 2000, meanSquaredLossFunction=False)
    listLossValid, listLossTrain, listEpochs = neuNet.train(trainingSet2, validationSet2)
    confMat, resNeuNet = neuNet.computeConfusionMatrix(testSet2)
    print("Neural Network,"  + str(neuronesHiddenLayer) + " neurones in hidden layer, etha = " + str(etha)
          + ", accuracy test set = " + str(resNeuNet))

    print(confMat)

    if plot:
        plt.plot(listEpochs, listLossValid, label="loss on validation set")
        plt.plot(listEpochs, listLossTrain, label="loss on training set")
        plt.xlabel("number of epochs")
        plt.ylabel("loss on training and validation sets")
        plt.title("evolution of the loss while training")
        plt.legend(bbox_to_anchor=(0.15, 0.9, 0.8, 0), loc='lower left',
                   ncol=2, mode="expand", borderaxespad=0.)
        plt.show()

#1.1
#plotSets()
#plotSets("")
#1.2.1
#testSimpleClassifier(linReg, 1)
#1.2.2
#testSimpleClassifier(logReg, 1)
#1.2.3
#testKnn(kNN, 1, nonLinearFeatures=False)
#testKnn(softKnn, 1, nonLinearFeatures=False)
#1.2.4
#testSimpleClassifier(perceptron, 1)
#1.3.1
#testKnn(kNN, "")
#testKnn(softKnn, "")
#1.3.2
#testOneVersusRest()
#1.4
#testSimpleClassifierWithNonLinearFeatures(logReg, 1)
#2
#testNeuralNetwork(True)



