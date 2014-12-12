from math import *


def evaluateRMSEGlobal(testSet,numTest,meanRating):
	totalSum = 0.0
	for user in testSet:
		for business in testSet[user]:
			totalSum += pow(meanRating - testSet[user][business][0],2)
	return sqrt(totalSum/numTest)


def evaluateRMSE(testSet,numTest):
	totalSum = 0.0
	totalMAE = 0.0
	for user in testSet:
		for business in testSet[user]:
			totalSum += pow(testSet[user][business][1] - testSet[user][business][0],2)
			totalMAE += abs(testSet[user][business][0] - testSet[user][business][1])
	return "RMSE = ",sqrt(totalSum/numTest),"MAE = ",totalMAE/numTest

def evaluateRMSEWeighted(testSet,numTest):
	totalSum = 0.0
	totalMAE = 0.0
	for user in testSet:
		for business in testSet[user]:
			totalSum += pow(testSet[user][business][2] - testSet[user][business][0],2)
			totalMAE += abs(testSet[user][business][0] - testSet[user][business][2])
	return "RMSE = ",sqrt(totalSum/numTest),"MAE = ",totalMAE/numTest

def calcMean(dict):
	sum = 0.0
	for val in dict.values():
		sum += val
	mean = sum/len(dict.values())
	return mean

def calcSum(dict):
	sum = 0.0
	for val in dict.values():
		sum += val
	return sum

def calcDataSparsity(trainingSet):
	non_zero_entries = 0
	for user in trainingSet:
		for bus in trainingSet[user]:
			non_zero_entries += 1
	data_sparsity = 1 - (non_zero_entries/(users*businesses)-non_zero_entries)
