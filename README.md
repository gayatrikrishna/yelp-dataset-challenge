cs760
=====

To run ARFFFilter.jar

java -jar ARFFFilter.jar <number_of_businesses> <level>

level can take one of these values :

0 - consider the businesses and the users who have rated for those businesses.
1 - level 0 + all those additional businesses that the users in the current set have rated for.
2 - level 1 + all those additional users who have rated for the businesses in the current set.

The current files in the repository are generated with the command
java -jar ARFFFilter.jar 2 1
