from sklearn import linear_model
from random import random
from Diagnostic.UI import *
import numpy as np
import math
def prediction(example, coef):
    s = 0.0
    for i in range(0, len(example)):
        s += coef[i] * example[i]
    return s
def sigmoidFunction(z):
    return 1.0 / (1.0 + math.exp(0.0 - z))
def cost_function(input, output, coef):
    noData = len(input)
    totalCost = 0.0
    for i in range(len(input)):
        example = input[i]
        predictedValue = sigmoidFunction(prediction(example, coef))
        realLabel = output[i]
        class1_cost = realLabel * math.log(predictedValue)
        class2_cost = (1 - realLabel) * math.log(1 - predictedValue)
        crtCost = - class1_cost - class2_cost
        totalCost += crtCost
    return totalCost / noData
def updateCoefs(input, output, coef, learningRate):
    #print(coef)
    noData = len(input)
    predictedValues = []
    realLabels = []
    for j in range(noData):
        crtExample = input[j]
        predictedValues.append(sigmoidFunction(prediction(crtExample, coef)))
        realLabels.append(output[j])
    #print("predictedValues",predictedValues)
    #print("realLabels",realLabels)
    for i in range(len(coef)):
        gradient = 0.0
        for j in range(noData):
             crtExample = input[j]
             #print(crtExample)
             gradient = gradient + crtExample[i] * (predictedValues[j] - realLabels[j])
        #print(gradient)
        #print(coef[i],"=",coef[i],"-",gradient,"*",learningRate)
        coef[i] = coef[i] - gradient * learningRate
    #print("Gradient",gradient)
    #print("Coef",coef)
    #print("\n")
    return coef
def train(input, output, learningRate, noIter):
    # print(input,output)
    coef = [0.5 for i in range(len(input[0]))]
    costs = []
    for it in range(noIter):
        coef = updateCoefs(input, output, coef, learningRate)
        crtCost = cost_function(input, output, coef)
        costs.append(crtCost)
    #print(coef)
    return costs, coef
def test(input, coef):
    predictedLabels = []
    print(input[0])
    for i in range(len(input)):
        predictedValue = sigmoidFunction(prediction(input[i], coef))
        print("The probability is",predictedValue)
        if (predictedValue >= 0.5):
            predictedLabels.append(1)
        else:
            predictedLabels.append(0)
    return predictedLabels
def accuracy(computedLabels, realLabels):
    noMatches = 0
    for i in range(len(computedLabels)):
        #print(computedLabels[i],realLabels[i])
        if (computedLabels[i] == realLabels[i]):
            noMatches += 1
    return noMatches / len(computedLabels)
def myLogisticRegression(input, output, learningRate, noIter):
    costs, coeficients = train(input, output, learningRate, noIter)
    computedLabels = test(input, coeficients)
    acc = accuracy(computedLabels, output)
    return acc
def SGDLogisticTool(x, y, learningRate, noEpoch):
    logreg = linear_model.LogisticRegression()
    logreg.max_iter = noEpoch
    logreg.fit (x, y)
    correct = sum(y == logreg.predict(x))
    return correct / len(x)

def LogisticSGD():
    fileI = "column_2C.dat"
    fileT = "test.dat"
    ui = UI(fileI, fileT)
    input=ui.getFromFile()[0]
    #input = [[-1.117488473, -0.771588515], [-0.768273325, -1.253831338], [-0.06984303,0.192897129], [0.628587266, 1.157382773], [1.327017562, 0.675139951]]
    #output = [1, 0, 1, 0, 1]
    output=ui.getFromFile()[1]
    lg=LG(input,output)
    input=lg.normalized_dataIn
    print("tool acc = ", SGDLogisticTool(input, output, 0.001, 10))
    print("my acc = ", myLogisticRegression(input, output, 0.001, 2))


LogisticSGD()
