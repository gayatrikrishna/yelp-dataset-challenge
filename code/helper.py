import arff
from metrics import *
# dictionary of ratings indexed by the business
# to calculate the business mean
def transpose(trainingSet):
	transposeTrainingSet = {}
	for userid in trainingSet:
		for businessid in trainingSet[userid]:
			if businessid not in transposeTrainingSet:
				transposeTrainingSet[businessid] = {}
			transposeTrainingSet[businessid][userid] = trainingSet[userid][businessid]
	return transposeTrainingSet


def loadFromArff(fileName,isTest):
	trainingSet = {}
	numTest = 0
	for row in arff.load(fileName):
		userid = row.user_id
		businessid = row.business_id
		numTest += 1
		if userid not in trainingSet:
			trainingSet[userid] = {}
		if(isTest):
			trainingSet[userid][businessid] = []
			trainingSet[userid][businessid].append(row.stars)
		else:
			trainingSet[userid][businessid] = row.stars
	return trainingSet,numTest

def getUserMean(trainingSet):
	userMean = {}
	for rec in trainingSet:
		businesses = trainingSet[rec]
		sum = calcSum(businesses)
		userMean[rec] = sum/len(businesses.keys())
	return userMean

def getItemMean(transposeTrainingSet):
	businessMean = {}
	for business in transposeTrainingSet:
		users = transposeTrainingSet[business]
		sum = calcSum(users)
		businessMean[business] = sum/len(users.keys())
	return businessMean

def getGlobalMean(trainingSet):
	userMeans = {}
	totalSum = 0.0
	numBusinesses = 0
	for rec in trainingSet:
		businesses = trainingSet[rec]
		sum = calcSum(businesses)
		totalSum += sum
		numBusinesses += len(businesses.keys())
		userMeans[rec] = sum/len(businesses.keys())
		#print rec,businesses,sum,"totalSum=",totalSum,userMeans[rec],len(businesses.keys())
	if numBusinesses == 0:
		return 0
	else:
		return userMeans,totalSum/numBusinesses
