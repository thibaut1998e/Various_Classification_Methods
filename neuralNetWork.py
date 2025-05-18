import numpy as np
import matplotlib.pyplot as plt

printLossWhileTraining = True



class  neuralNetWork:
    """caracterised by
    - An integer number of layers 'n'
    - An integer's list 'lengths' such that lengths[i] is the number of neurons from layer i.
    - A list of matrices 'weights' (n-1 matrices). Each matrix contains the weights of all connexions between 2 consecutive layers
    The size of weights[i] is (lengths[i]+1)*lengths[i+1]. The first line of the matrice is the bias

    - A list of layers 'layers'. Each layer is an array of size batchSize*lenghts[i].
    - A learning rate 'etha' which is a float number.
    - other training parameters e, t, nbMaxEpochs
    - meanSquaredLossFunction is a boolean which is true iff we want to use meanSquared error while training otherwise
    we use cross entropy
    """

    # The class constructor. The only arguments are the dimensions 'lengths' and the learning rate etha, as well as training parameters
    # as the weights and biaises are initialised randomly
    def __init__(self, lengths, etha, batchSize=2, e=10, t=0, nbMaxEpochs = 1000, meanSquaredLossFunction=False):
        self.numberOfLayers = len(lengths)
        self.lengths = lengths
        self.weights = [np.random.random(size = (lengths[c]+1, lengths[c+1]))*2-1 for c in range(self.numberOfLayers-1)]
        self.layers = [np.zeros((batchSize, self.lengths[i])) for i in range(self.numberOfLayers)]
        self.etha = etha/batchSize
        self.batchSize = batchSize
        self.e = e
        self.t = t
        self.nbMaxEpochs = nbMaxEpochs
        self.meanSquaredLossFunction = meanSquaredLossFunction



    def propagate(self, input):
        self.layers[0] = input

        for i in range(1, self.numberOfLayers):
            self.layers[i-1] = np.insert(self.layers[i-1],0,np.array([1]*len(self.layers[i-1])),axis=1)
            h = self.layers[i - 1].dot(self.weights[i - 1])
            if i != self.numberOfLayers-1:
                self.layers[i] = sigmoid(h)
            else:
                self.layers[i] = h
            print(self.layers[i])

        return self.layers[self.numberOfLayers - 1]


    def backPropagate(self, error):
        L = self.numberOfLayers
        deltaC =  error
        #if self.meanSquaredLossFunction:
            #deltaC *= self.layers[L-1] * (1 - self.layers[L-1])
        print("detaW1")
        print(np.transpose(self.layers[L-2]).dot(deltaC))
        self.weights[L-2] += self.etha * np.transpose(self.layers[L-2]).dot(deltaC)
        print("newW1")
        print(self.weights[L-2])
        for c in range(L-2,0,-1):

            deltaC = deltaC.dot(np.transpose(self.weights[c][1:])) * self.layers[c][:,1:] * (1 - self.layers[c][:,1:])
            print("deltaC0")
            print(deltaC)
            print("deltaW0")
            print(np.transpose(self.layers[c-1]).dot(deltaC))
            self.weights[c-1] += self.etha * np.transpose(self.layers[c-1]).dot(deltaC)
            print("newW0")
            print(self.weights[c-1])


    def iteration(self, input, expectedOutput):
        output = self.propagate(input)
        error = expectedOutput - output
        self.backPropagate(error)


    def train(self, trainingSet, validationSet):

        stop = False
        nbEpoch = 0
        oldLoss = self.getLoss(validationSet)
        listLossTrain = [self.getLoss(trainingSet)]
        listLossValid = [oldLoss]
        listEpochs = [0]
        while not stop and nbEpoch < self.nbMaxEpochs:
            for k in range(self.e):
                cpt = 0
                while cpt + self.batchSize <= len(trainingSet):
                    inputBatch = np.array([trainingSet[i].vector for i in range(cpt, cpt+self.batchSize)])
                    outputBatch = np.array([trainingSet[i].labelVector for i in range(cpt, cpt+self.batchSize)])
                    self.iteration(inputBatch, outputBatch)
                    cpt = cpt+self.batchSize
                nbEpoch += 1
            newLoss = self.getLoss(validationSet)
            if oldLoss - newLoss < self.t:
                stop = True
            oldLoss = newLoss
            listLossValid.append(oldLoss)
            listLossTrain.append(self.getLoss(trainingSet))
            listEpochs.append(nbEpoch)
            if printLossWhileTraining:
                print(oldLoss)
                print(self.getAccuracy(validationSet))
        return listLossValid, listLossTrain, listEpochs



    def getLoss(self, validationSet):
        sum = 0
        for x in validationSet:
            output = self.propagate(np.array([x.vector]))[0]
            for i in range(len(output)):
                if self.meanSquaredLossFunction:
                    sum += (x.labelVector[i] - output[i])**2
                else:
                    sum -= x.labelVector[i]*np.log(output[i]) + (1-x.labelVector[i])*np.log(1-output[i])
        return sum/len(validationSet)


    def predictedClass(self, x):
        y = self.propagate([x.vector])[0]
        besti = 0
        bestProba = 0
        for i in range(len(y)):
            if y[i] >= bestProba:
                besti = i
                bestProba = y[i]
        return besti


    def getAccuracy(self, validationSet):
        return self.computeConfusionMatrix(validationSet)[1]

    def computeConfusionMatrix(self, testSet):
        confMat = np.zeros((self.lengths[-1],self.lengths[-1]))
        for x in testSet:
            predictedClass = self.predictedClass(x)
            confMat[x.label][predictedClass] += 1
        accuracy = sum([confMat[i][i] for i in range(len(confMat))]) / len(testSet)

        return confMat, accuracy


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def normalizeSets(trainingSet, validationSet, testSet):
    listMean = [np.mean([x.vector[i] for x in trainingSet]) for i in range(len(trainingSet[0].vector))]
    listStdv = [np.std([x.vector[i] for x in trainingSet]) for i in range(len(trainingSet[0].vector))]
    for x in trainingSet:
        x.normalize(listMean, listStdv)
    for x in validationSet:
        x.normalize(listMean, listStdv)
    for x in testSet:
        x.normalize(listMean, listStdv)
    return listMean, listStdv

neuNet = neuralNetWork([2,3,1], 0.1, batchSize=1,  meanSquaredLossFunction=True)
neuNet.weights = [np.array([[0.1,0.2,0.3], [0.4,0.5,0.6], [0.7,0.8,0.9]]), np.array([[0.1], [0.2], [0.3], [0.4]])]
output = neuNet.propagate(np.array([[1,2]]))
error = 10 - output
print(error)
neuNet.backPropagate(error)