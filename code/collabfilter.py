import argparse
import arff
from math import *
from helper import *
from metrics import *
from similarity import *

class CollaborativeFilter:
	trainingSet = {}
	transposeTrainingSet = {}
	testSet = {}
	users = None
	businesses = None
	normalizedUserSet = {}
	normalizedItemSet = {}
	numTraining = 0
	numTest = 0
	meanRating = 0.0
cf = CollaborativeFilter

def loadFromFiles(args):
	cf.trainingSet,cf.numTraining = loadFromArff(args.train,False)
	cf.testSet,cf.numTest = loadFromArff(args.test,True)
	cf.transposeTrainingSet = transpose(cf.trainingSet)
	cf.normalizedUserSet = cf.trainingSet
	cf.normalizedItemSet = cf.transposeTrainingSet

def initialize():
	cf.businesses = set()
	cf.users = set(cf.trainingSet.keys())
	for user in cf.trainingSet:
		cf.businesses = cf.businesses | set(cf.trainingSet[user].keys())
	print "Data sparsity",calcDataSparsity()
	
def normalizeUserData():
	userMean = getUserMean(cf.normalizedUserSet)
	for user in cf.normalizedUserSet:
		for bus in cf.normalizedUserSet[user]:
			cf.normalizedUserSet[user][bus] = cf.normalizedUserSet[user][bus] - userMean[user]

def normalizeItemData():
	itemMean = getItemMean(cf.normalizedItemSet)
	for bus in cf.normalizedItemSet:
		for user in cf.normalizedItemSet[bus]:
			cf.normalizedItemSet[bus][user] = cf.normalizedItemSet[bus][user] - itemMean[bus]

def calcUserUserCF(k):
	#print cf.trainingSet
	#normalizeUserData()
	# get the global mean to subsititute for those users who are only in the test set but not in the training set.
	userMean,cf.meanRating = getGlobalMean(cf.trainingSet)
	#print "normalizedUserSet",cf.normalizedUserSet
	#print cf.meanRating,"number of users",len(cf.users),"entries in normalized user set",len(cf.normalizedUserSet)
	userDistances = userCosineDistance(cf.users,cf.trainingSet)#cf.normalizedUserSet)
	for user in cf.testSet:
		for bus in cf.testSet[user]:
			n = 0
			average = 0.0
			weighted_average = 0.0
			similarityValues = []
			if user not in userDistances:
				cf.testSet[user][bus].insert(1,cf.meanRating)
				cf.testSet[user][bus].insert(2,cf.meanRating)
			else:
				for nearestUser in sorted(userDistances[user], key=userDistances[user].get, reverse=True):
					if bus in cf.trainingSet[nearestUser]:
						average += cf.trainingSet[nearestUser][bus]
						weighted_average += userDistances[user][nearestUser]*cf.trainingSet[nearestUser][bus]
						similarityValues.append(userDistances[user][nearestUser])
						n = n + 1
						if n == k:
							break
				if n==0:
					#cf.testSet[user][bus].insert(1,cf.meanRating)
					cf.testSet[user][bus].insert(1,cf.meanRating)
					cf.testSet[user][bus].insert(2,cf.meanRating)
				else:
					cf.testSet[user][bus].insert(1,average/n)
					if sum(similarityValues) == 0:
						cf.testSet[user][bus].insert(2,cf.meanRating)
					else:
						cf.testSet[user][bus].insert(2,weighted_average/sum(similarityValues))
	print "RMSE with User-User Collaborative Filtering",evaluateRMSE(cf.testSet,cf.numTest)
	print "RMSE with User-User Collaborative Filtering - Weighted average",evaluateRMSEWeighted(cf.testSet,cf.numTest)

def calcItemItemCF(k):
	#normalizeItemData()
	businessDistances = itemCosineDistance(cf.businesses,cf.transposeTrainingSet)
	for user in cf.testSet:
		for bus in cf.testSet[user]:
			n = 0
			average = 0.0
			weighted_average = 0.0
			similarityValues = []
			# if the business is not present in the training set
			if bus not in businessDistances:
				#cf.testSet[user][bus].insert(1, cf.meanRating)
				cf.testSet[user][bus].insert(1,cf.meanRating)
				cf.testSet[user][bus].insert(2,cf.meanRating)
			else:
				for nearestBusiness in sorted(businessDistances[bus], key=businessDistances[bus].get, reverse=True):
					if user in cf.transposeTrainingSet[nearestBusiness]:
						average += cf.transposeTrainingSet[nearestBusiness][user]
						weighted_average += businessDistances[bus][nearestBusiness]*cf.transposeTrainingSet[nearestBusiness][user]
						similarityValues.append(businessDistances[bus][nearestBusiness])
						n += 1
						if n == k:
							break
				if n==0:
					#cf.testSet[user][bus].insert(1,cf.meanRating)
					cf.testSet[user][bus].insert(1,cf.meanRating)
					cf.testSet[user][bus].insert(2,cf.meanRating)
				else:
					cf.testSet[user][bus].insert(1,average/n)
					if sum(similarityValues) == 0:
						cf.testSet[user][bus].insert(2,cf.meanRating)
					else:
						cf.testSet[user][bus].insert(2,weighted_average/sum(similarityValues))
	print "Item-Item Collaborative Filtering",evaluateRMSE(cf.testSet,cf.numTest)
	print "RMSE with Item-Item Collaborative Filtering - Weighted average",evaluateRMSEWeighted(cf.testSet,cf.numTest)

def createParser():
	parser = argparse.ArgumentParser(description="Collaborative Filtering approaches")
	parser.add_argument("-r", default="reviews.arff", help="File having reviews")
	parser.add_argument("-train", default="training.arff", help="Training Set")
	parser.add_argument("-test", default="test.arff", help="Test Set")
	parser.add_argument("-k", default=4, help="Number of nearest neighbors")
	return parser.parse_args()

if __name__ == '__main__':
	cf.args = createParser()
	loadFromFiles(cf.args)
	initialize()
	calcUserUserCF(cf.args.k)
	calcItemItemCF(cf.args.k)