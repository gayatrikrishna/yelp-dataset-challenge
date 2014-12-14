import arff
import argparse
import random
import math
import sys
import matplotlib.pyplot as plt
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
	trainingIndex = []
	testIndex = []
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

def getIndexes(records):
	# sample 75% of the users
	users = set()
	items = set()
	for row in records:
		users.add(row.user_id)
		items.add(row.business_id)
	selection = int(math.floor(bl.args.train * len(users)))
	selectedUsers = random.sample(users,selection)
	selectedItems = random.sample(items,int(bl.args.testitempercent*len(items)))
	
	print "total number of records",len(records)
	print "total number of users", len(users)
	print "total number of items",len(items)
	print "total number of selected users",len(selectedUsers)
	print "total number of selected items",len(selectedItems)
	

	for r in xrange(0,bl.numRecords):
		if records[r].user_id in selectedUsers:
			bl.trainingIndex.append(r)
			bl.numTraining += 1
		else:
			if records[r].business_id in selectedItems:
				bl.testIndex.append(r)
				bl.numTest += 1
			else:
				bl.trainingIndex.append(r)
				bl.numTraining += 1
	
	print "Number of training instances",bl.numTraining
	print "Number of test instances",bl.numTest
	
	
	return bl.trainingIndex,bl.testIndex

def getSimpleSelection(records):
	users = set()
	items = set()
	for row in records:
		users.add(row.user_id)
		items.add(row.business_id)
	print "total number of users",len(users)
	print "total number of items",len(items)
	print "total number of entries",len(users)*len(items)
	total_entries = len(users)*len(items)
	zero_entries = total_entries - bl.numRecords
	print "Data sparsity = ", 1-float(bl.numRecords)/float(zero_entries)
	bl.numTraining = int(math.floor(bl.args.train * bl.numRecords))
	bl.numTest = bl.numRecords - bl.numTraining
	bl.trainingIndex = random.sample(range(bl.numRecords),bl.numTraining)
	bl.testIndex = [x for x in xrange(0,bl.numRecords) if x not in bl.trainingIndex]
	print "total number of instances",bl.numRecords
	print "number of training instances",bl.numTraining
	print "number of test instances",bl.numTest
	return bl.trainingIndex,bl.testIndex


def readArff(reviewFile,train):
	# read arff file into the training and test set
	records = list(arff.load(reviewFile))
	bl.numRecords = len(records)
	bl.trainingIndex, bl.testIndex = getSimpleSelection(records)#getIndexes(records)

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
	#print "RMSE of Test Set with Global Mean",evaluateRMSEGlobal(bl.testSet,bl.numTest,bl.meanRating)
	# calculate RMSE for user mean
	for user in bl.testSet:
		for bus in bl.testSet[user]:
			if user not in userMeans:
				bl.testSet[user][bus].insert(1,0.0)#bl.meanRating)
			else:
				bl.testSet[user][bus].insert(1,userMeans[user])
	#print bl.testSet
	print "RMSE of Test Set with User Mean",evaluateRMSE(bl.testSet,bl.numTest)

def calcBusinessMean():
	businessMeans = {}
	"""
	for business in bl.transposeTrainingSet:
		users = bl.transposeTrainingSet[business]
		sum = calcSum(users)
		businessMeans[business] = sum/len(users.keys())
	"""
	businessMeans = getItemMean(bl.transposeTrainingSet)
	for user in bl.testSet:
		for bus in bl.testSet[user]:
			if bus not in businessMeans:
				bl.testSet[user][bus].insert(1,0.0)
			else:
				bl.testSet[user][bus].insert(1,businessMeans[bus])
	#print "RMSE of Test Set with Business Mean",evaluateRMSE(bl.testSet,bl.numTest)

def plotHistogram(reviewFile):
	records = list(arff.load(reviewFile))
	bl.numRecords = len(records)
	numUserReviews = {}
	numBusReviews = {}
	
	users = set()
	items = set()
	for row in records:
		users.add(row.user_id)
		items.add(row.business_id)

	for user in users:
		numUserReviews[user] = 0
	for bus in items:
		numBusReviews[bus] = 0

	for row in xrange(0,bl.numRecords):
		userid = records[row].user_id
		businessid = records[row].business_id
		numUserReviews[userid] += 1
		numBusReviews[businessid] += 1

	print bl.numRecords
	print sum(numUserReviews.values())
	print sum(numBusReviews.values())
	# calculate a histogram
	xUserHist = [] # number of ratings per user
	yUserHist = [] # number of users
	xBusHist = [] # number of ratings per business
	yBusHist = [] # number of businesses
	cur_num = 0
	prev_val = bl.numRecords + 1
	#print sorted(numUserReviews,key=numUserReviews.get)
	for val in sorted(numUserReviews,key=numUserReviews.get):
		if numUserReviews[val] > prev_val:
			yUserHist.append(cur_num)
			xUserHist.append(prev_val)
			cur_num = 0
		cur_num += 1
		prev_val = numUserReviews[val]
	#print xHist,yHist
	yUserHist.append(cur_num)
	xUserHist.append(prev_val)

	prev_val = bl.numRecords + 1
	cur_num = 0
	for val in sorted(numBusReviews,key=numBusReviews.get):
		if numBusReviews[val] > prev_val:
			yBusHist.append(cur_num)
			xBusHist.append(prev_val)
			cur_num = 0
		cur_num += 1
		prev_val = numBusReviews[val]
	yBusHist.append(cur_num)
	xBusHist.append(prev_val)

	#print yUserHist,xUserHist
	#print yBusHist,xBusHist
	"""
	print "xuser"
	for x in xUserHist:
		print x
	print "yuser"
	for y in yUserHist:
		print y

	print "xbus"
	for x in xBusHist:
		print x
	print "ybus"
	for y in yBusHist:
		print y

	total = 0
	for i in xrange(0,len(xUserHist)):
		total += xUserHist[i] * yUserHist[i]
	print total
	total = 0
	for i in xrange(0,len(xBusHist)):
		total += xBusHist[i] * yBusHist[i]
	print total
	"""
	"""
	plt.hist(xUserHist,yUserHist,color='red')
	plt.show()
	"""

def createParser():
	parser = argparse.ArgumentParser(description="Baseline approaches for collaborative filtering")
	parser.add_argument("-r", default="reviews.arff", help="File having reviews")
	parser.add_argument("-u", default="users.arff", help="File having users")
	parser.add_argument("-b", default="businesses.arff", help="File having businesses")
	parser.add_argument("-train", default=0.7, help="training set percentage", type=float)
	parser.add_argument("-testitempercent", default=0.1, help="percentage of the item ratings to consider as test set", type=float)
	parser.add_argument("-histogram",default=False,help="plot an histogram",type=bool)
	return parser.parse_args()

if __name__ == '__main__':
	bl.args = createParser()
	if(bl.args.histogram):
		plotHistogram(bl.args.r)
	else:
		readArff(bl.args.r,bl.args.train)
		bl.transposeTrainingSet = transpose(bl.trainingSet)
		calcGlobalandUserMean()
		calcBusinessMean()
	

