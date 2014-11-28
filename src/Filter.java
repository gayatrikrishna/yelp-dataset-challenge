import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.core.converters.ArffSaver;

public class Filter {

	Instances filteredBusinessDataSet;
	Instances data;
	String businessFile = "data/yelp_academic_dataset_business.arff";
	String reviewFile = "data/yelp_academic_dataset_review.arff";
	String userFile = "data/yelp_academic_dataset_user.arff";
	// inverted index consisting of the list of all users who have reviewed a
	// business
	HashMap<String, List<String>> invertedIndexUser;
	// inverted index consisting of the list of all businesses that have been
	// reviewed by an user.
	HashMap<String, List<String>> invertedIndexBusiness;
	// Hashset of all users who reviewed a business.
	HashSet<String> users;
	// HashMap of businesses and instances
	HashMap<String, Instance> businessInstances;
	// HashMap of business column names (hash) and the corresponding business
	// ids.
	HashMap<String, String> businessHashMap;
	// reverse mapping
	HashMap<String, String> reverseBusinessMap;
	// Matrix of user and business reviews.
	HashMap<String, HashMap<String, Integer>> userBusinessRatings;
	// List of attributes to merge.
	ArrayList<Attribute> userAttributes;
	ArrayList<Attribute> businessHashAttributes;
	// Map of users to the corresponding instances.
	HashMap<String, Instance> userInstances;
	Instances filteredReviews;

	public void getBusinesses(int numBusinesses, int level) throws IOException {
		businessInstances = new HashMap<String, Instance>();
		businessHashMap = new HashMap<String, String>();
		reverseBusinessMap = new HashMap<String, String>();
		businessHashAttributes = new ArrayList<Attribute>();
		// read the business arff file
		try {
			BufferedReader reader = new BufferedReader(new FileReader(
					businessFile));
			// Read from ARFF format using weka reader.
			data = new Instances(reader);
			for (int i = 0; i < data.size(); ++i) {
				businessInstances.put(data.get(i).stringValue(15), data.get(i));
			}
			// TODO : Need to change it to random selection
			filteredBusinessDataSet = new Instances(data, 0, numBusinesses);
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		// write the filtered business set to an arff file.
		if (level == 0) {
			ArffSaver saver = new ArffSaver();
			saver.setInstances(filteredBusinessDataSet);
			saver.setFile(new File("./filteredBusinesses.arff"));
			saver.writeBatch();
		}
		// Load the business id list into a set.
		invertedIndexUser = new HashMap<String, List<String>>();
		for (int i = 0; i < filteredBusinessDataSet.size(); ++i) {
			invertedIndexUser.put(filteredBusinessDataSet.get(i)
					.stringValue(15), new ArrayList<String>());
		}

	}

	public void populateReviews(int level) throws IOException {
		invertedIndexBusiness = new HashMap<String, List<String>>();
		users = new HashSet<String>();
		userBusinessRatings = new HashMap<String, HashMap<String, Integer>>();

		Instances reviewData = null;
		try {
			BufferedReader reader = new BufferedReader(new FileReader(
					reviewFile));
			// Read from ARFF format using weka reader.
			reviewData = new Instances(reader);
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		filteredReviews = new Instances(reviewData, 0, 0);
		for (int i = 0; i < reviewData.size(); ++i) {
			String businessId = reviewData.get(i).stringValue(0);
			// populate the inverted index for business.
			String userId = reviewData.get(i).stringValue(4);

			if (!invertedIndexBusiness.containsKey(userId)) {
				invertedIndexBusiness.put(userId, new ArrayList<String>());
				invertedIndexBusiness.get(userId).add(businessId);
			} else {
				invertedIndexBusiness.get(userId).add(businessId);
			}

			if (invertedIndexUser.containsKey(businessId)) {
				invertedIndexUser.get(businessId).add(userId);
				filteredReviews.add(reviewData.get(i));
				users.add(userId);
			}

		}

		if (level == 0) {
			ArffSaver saver = new ArffSaver();
			saver.setInstances(filteredReviews);
			saver.setFile(new File("./filteredReviews.arff"));
			saver.writeBatch();
		}
		if (level == 1 || level == 2) {
			// take the set of all users and probe the inverted index business
			// to get other businesses
			// that user has rated.
			HashSet<String> uniqueBusinesses = new HashSet<String>();
			filteredReviews = new Instances(reviewData, 0, 0);
			filteredBusinessDataSet = new Instances(data, 0, 0);
			for (String user : users) {
				List<String> businesses = invertedIndexBusiness.get(user);
				for (String bus : businesses) {
					uniqueBusinesses.add(bus);
				}
			}
			for (String bus : uniqueBusinesses)
				filteredBusinessDataSet.add(businessInstances.get(bus));
			ArffSaver saver = new ArffSaver();
			saver.setInstances(filteredBusinessDataSet);
			saver.setFile(new File("./filteredBusinesses.arff"));
			saver.writeBatch();
		}
		if (level == 1) {
			// add additional rows for other business that the users have rated
			for (int i = 0; i < reviewData.size(); ++i) {
				if (users.contains(reviewData.get(i).stringValue(4)))
					filteredReviews.add(reviewData.get(i));
			}
		}
		if (level == 2) {
			HashSet<String> newSetBusinesses = new HashSet<String>();
			for (int i = 0; i < filteredBusinessDataSet.size(); ++i) {
				newSetBusinesses.add(filteredBusinessDataSet.get(i)
						.stringValue(15));
			}
			for (int i = 0; i < reviewData.size(); ++i) {
				if (newSetBusinesses.contains(reviewData.get(i).stringValue(0))) {
					filteredReviews.add(reviewData.get(i));
					if (!users.contains(reviewData.get(i).stringValue(4)))
						users.add(reviewData.get(i).stringValue(4));
				}
			}
		}
		ArffSaver saver = new ArffSaver();
		saver.setInstances(filteredReviews);
		saver.setFile(new File("./filteredReviews.arff"));
		saver.writeBatch();
	}

	public void populateUsers(int level) throws IOException {
		Instances userData = null;
		try {
			BufferedReader reader = new BufferedReader(new FileReader(userFile));
			// Read from ARFF format using weka reader.
			userData = new Instances(reader);
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		Instances filteredUsers = new Instances(userData, 0, 0);
		for (int i = 0; i < userData.size(); ++i) {
			if (users.contains(userData.get(i).stringValue(14)))
				filteredUsers.add(userData.get(i));
		}

		ArffSaver saver = new ArffSaver();
		saver.setInstances(filteredUsers);
		saver.setFile(new File("./filteredUsers.arff"));
		saver.writeBatch();

	}

	public void combineUserAndReviews() throws IOException {
		// Read both filtered reviews and users.
		Instances userData = null;
		Instances reviewData = null;
		userAttributes = new ArrayList<Attribute>();
		userInstances = new HashMap<String, Instance>();
		try {
			BufferedReader br = new BufferedReader(new FileReader(
					"./filteredUsers.arff"));
			userData = new Instances(br);
			br = new BufferedReader(new FileReader("./filteredReviews.arff"));
			reviewData = new Instances(br);
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		ArrayList<Attribute> allAttributes = new ArrayList<Attribute>();
		for (int i = 0; i < userData.numAttributes(); ++i) {
			Attribute attr = userData.attribute(i).copy(
					"user." + userData.attribute(i).name());
			if (i != 14) {
				allAttributes.add(attr);
			}
			userAttributes.add(attr);
		}
		for (int i = 0; i < reviewData.numAttributes(); ++i) {
			Attribute attr = reviewData.attribute(i).copy(
					"review." + reviewData.attribute(i).name());
			allAttributes.add(attr);
		}
		for (int i = 0; i < userData.size(); ++i) {
			userInstances.put(userData.get(i).stringValue(14), userData.get(i));
		}
		Instances combinedDataSet = new Instances("userReviews", allAttributes,
				reviewData.size());
		for (int i = 0; i < reviewData.size(); ++i) {
			Instance userInstance = userInstances.get(reviewData.get(i)
					.stringValue(4));
			Instance combinedInstance = userInstance.mergeInstance(reviewData
					.get(i));
			combinedInstance.deleteAttributeAt(14);
			combinedDataSet.add(combinedInstance);
		}
		ArffSaver saver = new ArffSaver();
		saver.setInstances(combinedDataSet);
		saver.setFile(new File("./UserReviewsData.arff"));
		saver.writeBatch();

	}

	public void generateUserReviewMatrix() throws IOException {
		for (int i = 0; i < filteredBusinessDataSet.size(); ++i) {
			businessHashMap.put("B" + i, filteredBusinessDataSet.get(i)
					.stringValue(15));
			reverseBusinessMap.put(
					filteredBusinessDataSet.get(i).stringValue(15), "B" + i);
			businessHashAttributes.add(new Attribute("B" + i));
		}
		// instantiate the user business ratings matrix only for users who have
		// rated the businesses.
		for (String userId : users) {
			userBusinessRatings.put(userId, new HashMap<String, Integer>());
		}
		// populate the user business ratings matrix.
		for (int i = 0; i < filteredReviews.size(); ++i) {
			String businessId = filteredReviews.get(i).stringValue(0);
			String userId = filteredReviews.get(i).stringValue(4);
			if (userBusinessRatings.containsKey(userId)
					&& reverseBusinessMap.containsKey(businessId)) {
				userBusinessRatings.get(userId).put(businessId,
						(int) filteredReviews.get(i).value(3));
			}
		}
		// generate the attribute set with users and business hash values.
		ArrayList<Attribute> allAttributes = new ArrayList<Attribute>();
		allAttributes.addAll(userAttributes);
		allAttributes.addAll(businessHashAttributes);
		Instances userReviewMatrix = new Instances("UserReviewMatrix",
				allAttributes, users.size());
		// System.out.println(userBusinessRatings);
		for (String userKey : userBusinessRatings.keySet()) {
			Instance userInstance = userInstances.get(userKey);
			Instance businessInstance = new SparseInstance(
					businessHashAttributes.size());

			for (int i = 0; i < businessHashAttributes.size(); ++i) {
				String attrName = businessHashAttributes.get(i).name();
				/*
				 * System.out.println(attrName);
				 * System.out.println(businessHashMap.get(attrName));
				 * System.out.println(userBusinessRatings.get(userKey));
				 */
				Integer rating = userBusinessRatings.get(userKey).get(
						businessHashMap.get(attrName));
				if (rating != null)
					businessInstance.setValue(i, rating);
			}
			Instance combinedInstance = userInstance
					.mergeInstance(businessInstance);
			userReviewMatrix.add(combinedInstance);
		}
		ArffSaver saver = new ArffSaver();
		saver.setInstances(userReviewMatrix);
		saver.setFile(new File("./UserReviewMatrix.arff"));
		saver.writeBatch();
		BufferedWriter bw = new BufferedWriter(new FileWriter(
				"BNum-BusinessID-Map"));
		for (String key : businessHashMap.keySet()) {
			bw.write(key + "," + businessHashMap.get(key) + "\n");
		}
		bw.close();
	}

}
