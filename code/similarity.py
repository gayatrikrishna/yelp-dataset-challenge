from math import *

def userPearsonCorrelation(user1,user2):
	numerator = 0.0
	denominator = 0.0
	user1_mod = 0.0
	user2_mod = 0.0
	for business_id in user2:
		user2_mod += user2[business_id] * user2[business_id]
	for business_id in user1:
		user1_mod += user1[business_id] * user1[business_id]
		if business_id in user2:
			numerator += (user1[business_id] * user2[business_id])
	denominator = sqrt(user1_mod) * sqrt(user2_mod)
	if denominator == 0:
		return 0.0
	else:
		return numerator/denominator

def itemPearsonCorrelation(item1,item2):
	numerator = 0.0
	denominator = 0.0
	item1_mod = 0.0
	item2_mod = 0.0
	for user_id in item2:
		item2_mod += item2[user_id] * item2[user_id]
	for user_id in item1:
		item1_mod += item1[user_id] * item1[user_id]
		if user_id in item2:
			numerator += (item1[user_id] * item2[user_id])
	denominator = sqrt(item1_mod) * sqrt(item2_mod)
	if denominator == 0:
		return 0.0
	else:
		return numerator/denominator

def userCosineDistance(users,dataSet):
	users = list(users)
	userDistances = {}
	for user1 in xrange(0,len(users)):
		userid1 = users[user1]
		userDistances[userid1] = {}
		for user2 in xrange(user1+1,len(users)):
			userid2 = users[user2]
			userDistances[userid1][userid2] = userPearsonCorrelation(dataSet[userid1],dataSet[userid2])
	#print userDistances
	return userDistances

def itemCosineDistance(businesses,dataSet):
	businesses = list(businesses)
	itemDistances = {}
	for business1 in xrange(0,len(businesses)):
		businessid1 = businesses[business1]
		itemDistances[businessid1] = {}
		for business2 in xrange(business1+1,len(businesses)):
			businessid2 = businesses[business2]
			itemDistances[businessid1][businessid2] = itemPearsonCorrelation(dataSet[businessid1],dataSet[businessid2])
	return itemDistances

