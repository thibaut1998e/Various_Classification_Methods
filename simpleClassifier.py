import numpy as np
import copy as cp
import random as rd
import matplotlib.pyplot as plt



printLossWhileTraining = True
linReg, logReg, perceptron = "LinReg", "LogReg", "Percepron"


def simgmoid(x):
    return 1/(1+np.exp(-x))

def heaviside(x):
    if x >= 0:
        return 1
    return 0


class simpleClassifier:
    def __init__(self, nbFeatures, batchSize, targetClass, etha, diff,  type = logReg, nbMaxEpochs = 1000):
        self.type = type
        self.targetClass = targetClass
        self.nbFeature = nbFeatures
        self.weights = np.random.random(size=(1, nbFeatures))
        self.biais = rd.random()
        self.etha = etha
        self.batchSize = batchSize
        self.nbMaxEpochs = nbMaxEpochs
        self.diff = diff


    def getOutPut(self, vector):
        h = np.vdot(self.weights, vector) + self.biais
        if self.type == linReg:
            return h
        if self.type == logReg:
            return simgmoid(h)
        if self.type == perceptron:
            return heaviside(h)

    def train(self, trainingSet, validationSet):
        if self.type != perceptron:
            improvement = 10000
            cpt = 0
            nbEpoch = 0
            oldLoss = self.cost(validationSet)
            while improvement > self.diff and nbEpoch < self.nbMaxEpochs:
                listInputs = [trainingSet[i].vector for i in range(cpt, cpt+self.batchSize)]
                listOutputs = [self.getOutPut(x) for x in listInputs]
                listTargets = [trainingSet[i].getValue(self.targetClass) for i in range(cpt, cpt+self.batchSize)]
                listErrors = [listOutputs[i] - listTargets[i] for i in range(len(listOutputs))]
                self.modifyWeights(listInputs, listErrors)
                cpt += self.batchSize
                if cpt + self.batchSize > len(trainingSet):
                    cpt = 0
                    nbEpoch += 1
                newLoss = self.cost(validationSet)
                improvement = oldLoss - newLoss
                if printLossWhileTraining:
                    print(newLoss)
                    print(self.getSucceedPercentage(validationSet))
                oldLoss = newLoss
            print("nb epoch")
            print(nbEpoch)
        else:
            bestParameters = cp.copy(self.weights), self.biais
            bestSucceedPercentage = self.getSucceedPercentage(validationSet)
            for k in range(self.nbMaxEpochs):
                for j in range(len(trainingSet)//self.batchSize):
                    listInputs = [trainingSet[i].vector for i in range(j*self.batchSize, (j+1) * self.batchSize)]
                    listOutputs = [self.getOutPut(x) for x in listInputs]
                    listTargets = [trainingSet[i].getValue(self.targetClass) for i in range(j*self.batchSize, (j+1) * self.batchSize)]
                    listErrors = [listOutputs[i] - listTargets[i] for i in range(len(listOutputs))]
                    self.modifyWeights(listInputs, listErrors)
                succeedPercentage = self.getSucceedPercentage(validationSet)
                if printLossWhileTraining:
                    print(succeedPercentage)
                if succeedPercentage > bestSucceedPercentage:
                    bestSucceedPercentage = succeedPercentage
                    bestParameters = cp.copy(self.weights), self.biais
            self.weights, self.biais = bestParameters



    def modifyWeights(self, listInputs, listErrors):
         inputs = np.array(listInputs)
         errors = np.array(listErrors)
         for k in range(self.nbFeature):

             self.weights[0][k] -= self.etha/self.batchSize * np.vdot(inputs[:,k], errors)
         self.biais -= self.etha/self.batchSize * sum(errors)




    def cost(self, validationSet):
        if self.type == linReg:
            return 0.5 * sum([(self.getOutPut(x.vector) - x.getValue(self.targetClass)) ** 2
                              for x in validationSet])
        if self.type == logReg:
            summ = 0
            for x in validationSet:
                y = self.getOutPut(x.vector)
                summ += x.getValue(self.targetClass) * np.log(y) + (1-x.getValue(self.targetClass)) * np.log(1-y)
            return -summ


    def getSucceedPercentage(self, validationSet):
        cpt = 0
        for x in validationSet:
            y = self.getOutPut(x.vector)
            decision = 0
            if y > 0.5:
                decision = 1
            if x.getValue(self.targetClass) == decision:
                cpt += 1
        return cpt / len(validationSet)

    #can be used only when there are two features
    def plotDecisionBoarder(self, color="black"):
        a, b, c = self.weights[0][0], self.weights[0][1], self.biais
        X = np.arange(-2.5, 2.5, 0.01)
        if self.type == linReg:
            Y = [-(a / b) * x + (0.5 - c) / b for x in X]
        if self.type == logReg or self.type == perceptron:
            Y = [-(a / b) * x - c / b for x in X]
        plt.plot(X, Y, c= color)




def addNonLinearFeatures(set):
    for x in set:
        x.addNonLinearFeatures()




def oneVersusRest(trainingSet, validationSet, testSet, batchSize, etha, nbOfClass=3):
    listClassifiers = [simpleClassifier(2, batchSize, targetClass, etha, 0.1, type = logReg) for targetClass in range(nbOfClass)]
    for classifier in listClassifiers:
        classifier.train(trainingSet, validationSet)
        classifier.plotDecisionBoarder()
    cpt = 0
    for x in testSet:
        listOutput = []
        for classifier in listClassifiers:
            listOutput.append(classifier.getOutPut(x.vector))
        bestClass = 0
        bestProba = 0
        for i in range(len(listOutput)):
            if listOutput[i] > bestProba:
                bestProba = listOutput[i]
                bestClass = i
        if bestClass == x.label:
            cpt += 1
    return cpt/len(validationSet)



classifier = simpleClassifier(2, 1, 0, 0.01, 1,  type = perceptron, nbMaxEpochs = 1000)
classifier.weights = np.array([[2,-1]])
classifier.biais = 2
classifier.plotDecisionBoarder()
plt.scatter([-2,2,1,1], [1,1,2,2.5], c = ["red","blue", "blue", "red"])
classifier.modifyWeights([np.array([1,3])], [1])
classifier.plotDecisionBoarder(color="yellow")
classifier = simpleClassifier(2, 1, 0, 1, 1,  type = perceptron, nbMaxEpochs = 1000)
classifier.weights = np.array([[2,-1]])
classifier.biais = 2
classifier.modifyWeights([np.array([1,3])], [1])
classifier.plotDecisionBoarder(color="green")
plt.grid()
plt.show()