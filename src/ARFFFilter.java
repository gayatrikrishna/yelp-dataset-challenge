import java.io.IOException;

public class ARFFFilter {

	
	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		// first argument - Number of businesses
		// second argument - Level 0, 1 or 2
		// Level 0 - retrieve only the businesses and the users who have rated these businesses
		// Level 1 - retrieve all the businesses which have been rated by the users who have rated these businesses
		// Level 2 - retrieve all the businesses (x) which have been rated by the users who have rated these businesses
		//         - and also the additional users who have rated the businesses x. 
		int numBusinesses = Integer.parseInt(args[0]);
		int level = Integer.parseInt(args[1]);
		Filter filter = new Filter();		
		filter.getBusinesses(numBusinesses,level);		
		filter.populateReviews(level);
		filter.populateUsers(level);
		filter.combineUserAndReviews();
		filter.generateUserReviewMatrix();
	}

}
