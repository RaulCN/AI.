#!/usr/bin/python
# -*- coding: utf-8 -*-

#corrigi os parenteses ma ainda não funciona
#fonte https://gist.githubusercontent.com/hzxie/a8db4f7449f7c91e266454ef310a51c5/raw/d393636c348f793d40a922b4f7d564638e58bb2e/Perceptron.py

from sys import argv
from sys import exit
from numpy import matrix
from datetime import datetime

def getSamplesWithLabels(sampleFilePath, labelFilePath, numberOfClasses):
    unlabelledSamples   = getSamples(sampleFilePath)
    labels              = getLabels(labelFilePath)
    samples             = [[] for i in range(0, numberOfClasses)]

    for index, sample in enumerate(unlabelledSamples):
        label = labels[index]
        samples[label].append(sample)

    return  samples

def getSamples(filePath):
    samples = []
    file    = open(filePath)

    for line in file.xreadlines():
        if line.strip() == '':
            continue

        sample = line.replace('\n', '').split(',')
        sample = [float(i) for i in sample]         # Covert str to float in list (named sample)
        samples.append(sample)

    return samples

def getLabels(filePath):
    labels  = []
    file    = open(filePath)

    for line in file.xreadlines():
        if line.strip() == '':
            continue

        label = int(line.replace('\n', ''))
        labels.append(label)

    return labels

def getPerceptronModels(samples, learningRate):
    weights         = []
    biases          = []
    numberOfClasses = len(samples)

    for i in range(0, numberOfClasses):
        print ('====================================================================')
        print ('Training Model #%d' % i)
        print ('====================================================================')
        positiveSamples = samples[i]
        negativeSamples = getOtherSamples(samples, i)
        weight, bias    = getPerceptronModel(positiveSamples, negativeSamples, learningRate)

        weights.append(weight)
        biases.append(bias)

    return weights, biases

def getOtherSamples(samples, exceptClassIndex):
    mergedSamples = []

    for index, samplesHavingSameLabel in enumerate(samples):
        if index == exceptClassIndex:
            continue
        for sample in samplesHavingSameLabel:
            mergedSamples.append(sample)

    return mergedSamples

def getPerceptronModel(positiveSamples, negativeSamples, learningRate):
    if len(positiveSamples) == 0 or len(negativeSamples) == 0:
        raise Exception("There's no sample available.")

    numberOfFeatures    = len(positiveSamples[0])
    weight              = [1 for i in range(0, numberOfFeatures)]
    bias                = 1

    minCostWeight       = [1 for i in range(0, numberOfFeatures)]
    minCostBias          = 1
    minCost             = len(positiveSamples) + len(negativeSamples)
    iterationTimes      = 0

    while True:
        cost = 0
        for sample in positiveSamples:
            if getPredictedLabel(weight, bias, sample) < 0:
                cost   += 1
                bias   += learningRate

                for i in range(0, numberOfFeatures):
                    weight[i] += learningRate * sample[i]

        for sample in negativeSamples:
            if getPredictedLabel(weight, bias, sample) > 0:
                cost   += 1
                bias   -= learningRate

                for i in range(0, numberOfFeatures):
                    weight[i] -= learningRate * sample[i]

        if cost < minCost:
            minCost         = cost
            minCostWeight   = [weight[i] for i in range(0, numberOfFeatures)]
            minCostBias     = bias

        if cost == 0: # For linear separable data
        # if iterationTimes == 50:
            print ('[DEBUG] weights = %s') % minCostWeight
            print ('[DEBUG] bias = %s') % minCostBias
            return minCostWeight, minCostBias
        else:
            iterationTimes += 1
            print ('[DEBUG] #%d: cost = %d, positive samples = %d, negative samples = %d') % (iterationTimes, cost, len(positiveSamples), len(negativeSamples))

def getPredictedLabel(weight, bias, sample):
    return (matrix(weight) * matrix(sample).T).item(0) + bias;

def getClassificationAccuracy(testingSamples, weights, biases):
    numberOfClasses                  = len(biases)
    totalSamples                     = [0] * numberOfClasses
    numberOfMisclassificationSamples = [0] * numberOfClasses

    for groupId, samplesHavingSameLabel in enumerate(testingSamples):
        weight                  = weights[groupId]
        bias                    = biases[groupId]
        totalSamples[groupId]  += len(samplesHavingSameLabel)

        for sample in samplesHavingSameLabel:
            if getPredictedLabel(weight, bias, sample) < 0:
                numberOfMisclassificationSamples[groupId] += 1

        print ('[INFO] Total Samples in #%d: %d') % (groupId, totalSamples[groupId])
        print ('[INFO] Number of misclassification samples in #%d = %d') % (groupId, numberOfMisclassificationSamples[groupId])

    return 100 - sum(numberOfMisclassificationSamples) * 100.0 / sum(totalSamples)

def main():
    if len(argv) != 7:
        print ('Usage: python Perceptron.py TrainingSamplePath TrainingLabelPath TestingSamplePath TestingLabelPath numberOfClasses learningRate')
        exit()

    print ('[INFO] Start Time: %s.') % datetime.now()
    trainingSamplePath      = argv[1]
    trainingLabelPath       = argv[2]
    testingSamplePath       = argv[3]
    testingLabelPath        = argv[4]
    numberOfClasses         = int(argv[5])
    learningRate            = float(argv[6])

    trainingSamples         = getSamplesWithLabels(trainingSamplePath, trainingLabelPath, numberOfClasses)
    testingSamples          = getSamplesWithLabels(testingSamplePath, testingLabelPath, numberOfClasses)
    weights, biases         = getPerceptronModels(trainingSamples, learningRate)

    classificationAccuracy  = getClassificationAccuracy(testingSamples, weights, biases)
    print ('[INFO] The classification accuracy is %.2f%%') % classificationAccuracy

    print ('[INFO] End Time: %s.') % datetime.now()

if __name__ == "__main__":
    main()
