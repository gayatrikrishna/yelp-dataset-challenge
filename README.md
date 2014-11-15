cs760 Yelp Dataset Challenge Project
=====

###Cleaning Data:
The data started in JSON format, but this wasn't usable; the process to convert
JSON objects into usable, accurate ARFF formatted files was a non-trivial task.

Part 1-Convert from nested JSON objects to flattened CSV files
Part 2-Clean CSV data & further flatten nested data where needed to get the data
compatible with the ARFF format (this part took the longest)
Part 3-Replace empty values in cases where the data is in fact known where a blank 
represents an implied false value.

Removing sparsity from the data where possible:
We avoided unknown data where possible during the cleaning process.

### Users (unknown parameters):
For features such as compliments, it is clear the non-presence of a count meant 
a zero was implied. We fixed this so an actual zero was included.

### Business (unknown parameters):
Attributes: if we don't know whether a business has bathrooms or accepts credit
cards, not knowing this is true doesn't necessarily tell us anything. Nothing
could be done in cases like this (the value remains unkonwn unless it's 
explicitly included). 
Categories: We turned categories into flat binary attributes where only a true 
or false value is possible. 
Neighborhoods: a similar approach was applied here to flatten this data into 
many columns of binary attributes.

### Checkins Tip & Review: 
Tips & Reviews had no unknown values. In the case of check-ins we know it's 
possible to visit a business without "checking in" so no assumptions could be 
made with this data (empty values were converted to unknowns "?" in ARFF)
