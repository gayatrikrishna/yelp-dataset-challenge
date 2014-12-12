import numpy as np
import argparse
from helper import *
from scipy.linalg import *

class SVD:
	args = None
	# matrix of users and ratings
	M = None
	Mfinal = None
	# decomposed matrices
	U = None
	V = None
	numUsers = 0
	numItems = 0
	userMap = {}
	busMap = {}
	trainingSet = {}
	testSet = {}
	averageUserMap = {}
	averageBusMap = {}
	rowMean = {}
	Ared = None
	numTraining = 0
	numTest = 0
sv = SVD


def initialize():
	i = 0
	j = 0
	businesses = set()
	users = set(sv.trainingSet.keys())
	for user in users:
		sv.userMap[user] = i
		sv.averageUserMap[user] = 0
		i += 1
	for user in sv.trainingSet:
		businesses = businesses | set(sv.trainingSet[user].keys())
	for bus in businesses:
		sv.busMap[bus] = j
		sv.averageBusMap[bus] = 0
		j += 1
	#print sv.userMap,sv.busMap
	# have a mapping from userid and business id to an integer 
	# so as to create a matrix
	sv.numUsers = len(users)
	sv.numItems = len(businesses)

	sv.M = [[None]*sv.numItems for x in range(sv.numUsers)]
	sv.averageUserMap = [0 for user in range(sv.numUsers)]
	sv.averageBusMap = [0 for bus in range(sv.numItems)]
	
	for user in sv.trainingSet:
		user_id = sv.userMap[user]
		for bus in sv.trainingSet[user]:
			business_id = sv.busMap[bus]
			sv.M[user_id][business_id] = sv.trainingSet[user][bus]
			sv.averageUserMap[user_id] += sv.trainingSet[user][bus]
			sv.averageBusMap[business_id] += sv.trainingSet[user][bus]

	for user in sv.userMap:
		user_id = sv.userMap[user]
		sv.averageUserMap[user_id] = float(sv.averageUserMap[user_id])/sv.numUsers
	for bus in sv.busMap:
		bus_id = sv.busMap[bus]
		sv.averageBusMap[bus_id] = float(sv.averageBusMap[bus_id])/sv.numItems
	print sv.averageBusMap
	imputeMissingData()

# imputes the missing data taking the row and column averages
def imputeMissingData():
	for i in xrange(0,sv.numUsers):
		sv.rowMean[i] = 0.0
		for j in xrange(0,sv.numItems):
			# impute missing values with column average (item average)
			#print "i=",i,"j=",j,"M[i][j]=",sv.M[i][j]
			if sv.M[i][j] == None:
				sv.M[i][j] = sv.averageBusMap[j]
			sv.rowMean[i] += sv.M[i][j]
		sv.rowMean[i] = float(sv.rowMean[i]/sv.numUsers)
	# obtain row centered matrix by subtracting the mean from each entry (normalizing)
	for i in xrange(0,sv.numUsers):
		for j in xrange(0,sv.numItems):
			sv.M[i][j] = sv.M[i][j] - sv.rowMean[i]
	print sv.M
	sv.Mfinal = np.matrix(sv.M)

def loadIntoMatrix(fileName):
	sv.trainingSet,sv.numTraining = loadFromArff(sv.args.train,False)
	sv.testSet,sv.numTest = loadFromArff(sv.args.test,True)	

def computeSVD():
	imputeMissingData()
	(U, S, V) = svd(sv.Mfinal)
	total = np.sum(S)
	k = total * sv.args.factor
	e = 0
	c = 0
	while e < k:
		e += S[c]
		c += 1
	# get the reduced matrices
	Ured = np.matrix(np.copy(U)[:,0:c])
	Sred = np.copy(S)[0:c]
	Vred = np.matrix(np.copy(V)[0:c,:])

	sv.Ared = Ured * np.diag(Sred) * Vred

def evaluateOnTest():
	userMean,meanRating = getGlobalMean(sv.trainingSet)
	for user in sv.testSet:
		if user in sv.userMap:
			i = sv.userMap[user]
		for bus in sv.testSet[user]:
			if bus in sv.busMap:
				j = sv.busMap[bus]
			if user not in sv.userMap or bus not in sv.busMap:
				sv.testSet[user][bus].insert(1,meanRating)
			else:
				sv.testSet[user][bus].insert(1,sv.rowMean[sv.userMap[user]]+sv.Ared[i,j])
	print evaluateRMSE(sv.testSet,sv.numTest)

def createParser():
	parser = argparse.ArgumentParser(description="Singular value Decomposition")
	parser.add_argument("-train", default="training.arff", help="Training file")
	parser.add_argument("-test", default="test.arff", help="Test file")
	parser.add_argument("-factor", default=0.8, help="Latent factors in terms of percentage")
	return parser.parse_args()


if __name__ == '__main__':
	sv.args = createParser()
	loadIntoMatrix(sv.args)
	initialize()
	computeSVD()
	evaluateOnTest()

