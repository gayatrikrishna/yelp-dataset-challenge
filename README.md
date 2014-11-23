cs760 Yelp Dataset Challenge Project
=====

To run ARFFFilter.jar

java -jar ARFFFilter.jar <number_of_businesses> <level>

level can take one of these values :

0 - consider the businesses and the users who have rated for those businesses.
1 - level 0 + all those additional businesses that the users in the current set have rated for.
2 - level 1 + all those additional users who have rated for the businesses in the current set.

The current files in the repository are generated with the command
java -jar ARFFFilter.jar 2 1



The data started in JSON format, but this wasn't usable; the process to convert
JSON objects into usable, accurate ARFF formatted files was a non-trivial task.


Part 2-Clean CSV data & further flatten nested data where needed to get the data
compatible with the ARFF format (this part took the longest)
Part 3-Replace empty values in cases where the data is in fact known where a blank 
represents an implied false value.


We avoided unknown data where possible during the cleaning process.





### Business (unknown parameters):
Attributes: if we don't know whether a business has bathrooms or accepts credit
cards, not knowing this is true doesn't necessarily tell us anything. Nothing

explicitly included). 
Categories: We turned categories into flat binary attributes where only a true 
or false value is possible. 
Neighborhoods: a similar approach was applied here to flatten this data into 
many columns of binary attributes.

### Checkins Tip & Review: 
Tips & Reviews had no unknown values. In the case of check-ins we know it's 
possible to visit a business without "checking in" so no assumptions could be 
made with this data (empty values were converted to unknowns "?" in ARFF)

##Clustering Data For Collaborative Filtering:
By discovering groupings among users we may be able to devise more accurate
review predictors. For example, some people might like fast food chains, 
others might be vegans, or prefer specific genres, or there may be other factors
where it might make natural sense to group these people together. For example,

various attributes of a business might be appealing to users. Past research has
demonstrated that clustering can significantly enhance predictive accuracy.

K-Means clustering was chosen because it can be effective with high dimensional
sparse data matrices. For example, businesses have nearly 1,000 features, and
most businesses have very little data regarding these features (generally less
than 3% or 4% of the features are populated with data).
