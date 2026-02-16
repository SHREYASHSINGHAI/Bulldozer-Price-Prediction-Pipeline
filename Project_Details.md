PROJECT DETAILS

PROBLEM STATEMENT
How well we can predict the future sale prices of bulldozers,
given its characteristics and examples of how much similar bulldozers have been sold for.


DATA
The dataset is downloaded from Kaggle bluebook for Bulldozers competition : https://www.kaggle.com/competitions/bluebook-for-bulldozers/data
There are three main datasets:
Train.csv is the training set, which contains data through the end of 2011.
Valid.csv is the validation set, which contains data from January 1, 2012 - April 30, 2012 You make predictions on this set.
Test.csv is the test set, it contains data from May 1, 2012 - November 2012.

The key fields are in train.csv are:
SalesID: the uniue identifier of the sale
MachineID: the unique identifier of a machine.  A machine can be sold multiple times
saleprice: what the machine sold for at auction (only provided in train.csv)
saledate: the date of the sale


EVALUATION
The evaluation metric for this competition is the RMSLE (root mean squared log error) between the actual and predicted auction prices.