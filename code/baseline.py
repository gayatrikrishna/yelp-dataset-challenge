import arff
import argparse
import random
import math
import sys
from metrics import *
from helper import *

# compute baselines, global mean, user mean, item mean

class BaseLine:
	args = None
	trainingSet = {}
	testSet = {}
	numRecords = 0
	numTraining = 0
	numTest = 0
	trainingIndex = None
	testIndex = None
	meanRating = 0.0
	transposeTrainingSet = {}
	trainingFile = None
	testFile = None
bl = BaseLine


def createNewFile(fileHandle):
	fileHandle.write("@relation yelp_academic_dataset_review_training\n")
	fileHandle.write("@attribute business_id string\n")
	fileHandle.write("@attribute user_id string\n")
	fileHandle.write("@attribute stars numeric\n")
	fileHandle.write("\n")
	fileHandle.write("@data\n")
	return fileHandle

def readArff(reviewFile,train):
	# read arff file into the training and test set
	records = list(arff.load(reviewFile))
	bl.numRecords = len(records)
	bl.numTraining = int(math.floor(train * bl.numRecords))
	bl.numTest = bl.numRecords - bl.numTraining
	bl.trainingIndex = random.sample(range(bl.numRecords),bl.numTraining)
	
	bl.testIndex = [x for x in xrange(0,bl.numRecords) if x not in bl.trainingIndex]

	# write to training and test set files
	trainingFile = open("training.arff","w")
	trainingFile = createNewFile(trainingFile)
	testFile = open("test.arff","w")
	testFile = createNewFile(testFile)
	
	for row in xrange(0,bl.numRecords):
		userid = records[row].user_id
		businessid = records[row].business_id
		if row in bl.trainingIndex:	
			if userid not in bl.trainingSet:
				bl.trainingSet[userid] = {}	
			bl.trainingSet[userid][businessid] = records[row].stars
			trainingFile.write(businessid+","+userid+","+str(records[row].stars)+"\n")
		else:
			if userid not in bl.testSet:
				bl.testSet[userid] = {}
			bl.testSet[userid][businessid] = []
			bl.testSet[userid][businessid].append(records[row].stars)
			testFile.write(businessid+","+userid+","+str(records[row].stars)+"\n")
	#print bl.testSet

def calcGlobalandUserMean():
	userMeans = {}
	userMeans,bl.meanRating = getGlobalMean(bl.trainingSet)
	print bl.meanRating
	#print bl.meanRating
	print "RMSE of Test Set with Global Mean",evaluateRMSEGlobal(bl.testSet,bl.numTest,bl.meanRating)
	# calculate RMSE for user mean
	for user in bl.testSet:
		for bus in bl.testSet[user]:
			if user not in userMeans:
				bl.testSet[user][bus].insert(1,bl.meanRating)
			else:
				bl.testSet[user][bus].insert(1,userMeans[user])
	#print bl.testSet
	print "RMSE of Test Set with User Mean",evaluateRMSE(bl.testSet,bl.numTest)

def calcBusinessMean():
	businessMeans = {}
	for business in bl.transposeTrainingSet:
		users = bl.transposeTrainingSet[business]
		sum = calcSum(users)
		businessMeans[business] = sum/len(users.keys())
	for user in bl.testSet:
		for bus in bl.testSet[user]:
			if bus not in businessMeans:
				bl.testSet[user][bus].insert(1,0.0)
			else:
				bl.testSet[user][bus].insert(1,businessMeans[bus])
	print "RMSE of Test Set with Business Mean",evaluateRMSE(bl.testSet,bl.numTest)

def createParser():
	parser = argparse.ArgumentParser(description="Baseline approaches for collaborative filtering")
	parser.add_argument("-r", default="reviews.arff", help="File having reviews")
	parser.add_argument("-u", default="users.arff", help="File having users")
	parser.add_argument("-b", default="businesses.arff", help="File having businesses")
	parser.add_argument("-train", default=0.5, help="training set percentage", type=float)
	return parser.parse_args()

if __name__ == '__main__':
	bl.args = createParser()
	readArff(bl.args.r,bl.args.train)
	bl.transposeTrainingSet = transpose(bl.trainingSet)
	calcGlobalandUserMean()
	calcBusinessMean()
	

