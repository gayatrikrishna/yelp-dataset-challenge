import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;


public class Filter {
	
	Instances filteredBusinessDataSet;
	Instances data;
	String businessFile = "yelp_academic_dataset_business.arff";
	String reviewFile = "yelp_academic_dataset_review.arff";
	String userFile = "yelp_academic_dataset_user.arff";
	// inverted index consisting of the list of all users who have reviewed a business
	HashMap<String,List<String>> invertedIndexUser;
	// inverted index consisting of the list of all businesses that have been reviewed by an user.
	HashMap<String,List<String>> invertedIndexBusiness;
	// Hashset of all users who reviewed a business.
	HashSet<String> users;
	// HashMap of businesses and instances
	HashMap<String,Instance> businessInstances;
	
	public void getBusinesses(int numBusinesses,int level) throws IOException{
		businessInstances = new HashMap<String,Instance>();
		// read the business arff file
		try {
			BufferedReader reader = new BufferedReader(new FileReader(businessFile));
			// Read from ARFF format using weka reader.
			data = new Instances(reader);
			for(int i = 0; i < data.size(); ++i){
				businessInstances.put(data.get(i).stringValue(15), data.get(i));
			}
			// TODO : Need to change it to random selection 
			filteredBusinessDataSet = new Instances(data,0,numBusinesses);			
			reader.close();
		}
		catch(IOException e){
			e.printStackTrace();
		}	
		// write the filtered business set to an arff file.
		if(level == 0){
			ArffSaver saver = new ArffSaver();
			saver.setInstances(filteredBusinessDataSet);
			saver.setFile(new File("./filteredBusinesses.arff"));
			saver.writeBatch();
		}
		// Load the business id list into a set.
		invertedIndexUser = new HashMap<String,List<String>>();		
		for(int i = 0; i < filteredBusinessDataSet.size(); ++i){
			invertedIndexUser.put(filteredBusinessDataSet.get(i).stringValue(15),new ArrayList<String>());			
		}
		
	}
	
	public void populateReviews(int level) throws IOException{
		invertedIndexBusiness = new HashMap<String,List<String>>();
		users = new HashSet<String>();
		
		Instances reviewData = null;
		try {
			BufferedReader reader = new BufferedReader(new FileReader(reviewFile));
			// Read from ARFF format using weka reader.
			reviewData = new Instances(reader);					
			reader.close();
		}
		catch(IOException e){
			e.printStackTrace();
		}		
		Instances filteredReviews = new Instances(reviewData,0,0);
		for(int i = 0; i < reviewData.size(); ++i){							
			String businessId = reviewData.get(i).stringValue(0);
			// populate the inverted index for business.						
			String userId = reviewData.get(i).stringValue(4);		
			
			if(!invertedIndexBusiness.containsKey(userId)){
				invertedIndexBusiness.put(userId, new ArrayList<String>());
				invertedIndexBusiness.get(userId).add(businessId);
			}
			else
			{
				invertedIndexBusiness.get(userId).add(businessId);
			}
			
			if(invertedIndexUser.containsKey(businessId)) {
				invertedIndexUser.get(businessId).add(userId);
				filteredReviews.add(reviewData.get(i));
				users.add(userId);
			}
				
		}
		if(level == 0){
			ArffSaver saver = new ArffSaver();
			saver.setInstances(filteredReviews);
			saver.setFile(new File("./filteredReviews.arff"));
			saver.writeBatch();
		}
		if(level == 1 || level == 2){
			// take the set of all users and probe the inverted index business to get other businesses
			// that user has rated.
			filteredBusinessDataSet = new Instances(data,0,0);
			for(String user : users){
				List<String> businesses = invertedIndexBusiness.get(user);
				for(String bus : businesses){
					filteredBusinessDataSet.add(businessInstances.get(bus));
				}
			}
			ArffSaver saver = new ArffSaver();
			saver.setInstances(filteredBusinessDataSet);
			saver.setFile(new File("./filteredBusinesses.arff"));
			saver.writeBatch();
		}
		if(level == 1){
			// add additional rows for other business that the users have rated
			for(int i = 0; i < reviewData.size(); ++i){
				if(users.contains(reviewData.get(i).stringValue(4)))
						filteredReviews.add(reviewData.get(i));
			}			
		}
		if(level == 2){
			HashSet<String> newSetBusinesses = new HashSet<String>();
			for(int i = 0; i < filteredBusinessDataSet.size(); ++i){
				newSetBusinesses.add(filteredBusinessDataSet.get(i).stringValue(15));
			}
			for(int i = 0; i < reviewData.size(); ++i){
				if(newSetBusinesses.contains(reviewData.get(i).stringValue(0))) {
						filteredReviews.add(reviewData.get(i));
						if(!users.contains(reviewData.get(i).stringValue(4)))
								users.add(reviewData.get(i).stringValue(4));
				}
			}
		}
		ArffSaver saver = new ArffSaver();
		saver.setInstances(filteredReviews);
		saver.setFile(new File("./filteredReviews.arff"));
		saver.writeBatch();
	}
	
	public void populateUsers(int level) throws IOException{
		Instances userData = null;
		try {
			BufferedReader reader = new BufferedReader(new FileReader(userFile));
			// Read from ARFF format using weka reader.
			userData = new Instances(reader);					
			reader.close();
		}
		catch(IOException e){
			e.printStackTrace();
		}		
		Instances filteredUsers = new Instances(userData,0,0);
		for(int i = 0; i < userData.size(); ++i){
			if(users.contains(userData.get(i).stringValue(14)))
					filteredUsers.add(userData.get(i));
		}
		
		ArffSaver saver = new ArffSaver();
		saver.setInstances(filteredUsers);
		saver.setFile(new File("./filteredUsers.arff"));
		saver.writeBatch();
	
		
	}
	
	public void combineUserAndReviews() throws IOException{
		// Read both filtered reviews and users.
		Instances userData = null;
		Instances reviewData = null;
		HashMap<String,Instance> userInstances = new HashMap<String,Instance>();
		try{
			BufferedReader br = new BufferedReader(new FileReader("./filteredUsers.arff"));
			userData = new Instances(br);			
			br = new BufferedReader(new FileReader("./filteredReviews.arff"));
			reviewData = new Instances(br);
			br.close();		
		}
		catch(IOException e){
			e.printStackTrace();
		}
		ArrayList<Attribute> allAttributes = new ArrayList<Attribute>();
		for(int i = 0; i < userData.numAttributes(); ++i){
			if(i != 14){
				Attribute attr = userData.attribute(i).copy("user."+userData.attribute(i).name());				
				allAttributes.add(attr);		
			}
		}
		for(int i = 0; i < reviewData.numAttributes(); ++i){
			Attribute attr = reviewData.attribute(i).copy("review."+reviewData.attribute(i).name());
			allAttributes.add(attr);
		}
		for(int i = 0; i < userData.size(); ++i){
			userInstances.put(userData.get(i).stringValue(14), userData.get(i));
		}
		Instances combinedDataSet = new Instances("userReviews", allAttributes, reviewData.size());
		for(int i = 0; i < reviewData.size(); ++i){			
			Instance userInstance = userInstances.get(reviewData.get(i).stringValue(4));
			Instance combinedInstance = userInstance.mergeInstance(reviewData.get(i));			
			combinedInstance.deleteAttributeAt(14);
			combinedDataSet.add(combinedInstance);
		}
		ArffSaver saver = new ArffSaver();
		saver.setInstances(combinedDataSet);
		saver.setFile(new File("./UserReviewsData.arff"));
		saver.writeBatch();
		
	}

}
