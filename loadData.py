from sklearn.datasets import make_blobs
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import copy as cp

nbOfClasses = 3

class labelledData:
    def __init__(self, vector, label):
        self.vector = vector
        self.label = label
        labelVector = [0 for i in range(nbOfClasses)]
        labelVector[label] = 1
        self.labelVector = np.array(labelVector)

    def addNonLinearFeatures(self):
        self.vector = np.array(list(self.vector) + [self.vector[0]**2, self.vector[1]**2, self.vector[0]*self.vector[1]])

    def normalize(self, listMean, listStdDev):
        for j in range(len(self.vector)):
            self.vector[j] = (self.vector[j] - listMean[j]) / listStdDev[j]

    def getValue(self, targetClass=""):
        if targetClass == "":
            return self.label
        return self.labelVector[targetClass]



def createSets():
    X, t = make_blobs(n_samples=[400,800,400], centers=[[0,0],[1,2],[2,3]], n_features=2, random_state=2019)
    indices = np.arange(X.shape[0])
    rd.seed(2020)
    rd.shuffle(indices)
    #X_train = X[indices[:100], :]
    X_train = X[indices[:800],:]
    X_val = X[indices[800:1200],:]
    X_test = X[indices[1200:],:]
    t_train = t[indices[:800]]
    t_val = t[indices[800:1200]]
    t_test = t[indices[1200:]]
    trainingSet = [labelledData(X_train[i], t_train[i]) for i in range(len(X_train))]
    validationSet = [labelledData(X_val[i], t_val[i]) for i in range(len(X_val))]
    testSet = [labelledData(X_test[i], t_test[i]) for i in range(len(X_test))]
    return trainingSet, validationSet, testSet

colors = ["blue", "red", "yellow"]


def plotSet(set, targetClass=1):
    if targetClass == "":
        for i in range(len(set)):
            plt.scatter(set[i].vector[0], set[i].vector[1], c=colors[set[i].label])
        plt.title("training set all classes")
    else:
        for i in range(len(set)):
            plt.scatter(set[i].vector[0], set[i].vector[1], c=colors[set[i].labelVector[targetClass]])
        plt.title("training set class " + str(targetClass) + " VS others")

    plt.xlabel("feature 1")
    plt.ylabel("feature 2")










