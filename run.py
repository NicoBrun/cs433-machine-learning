import implementations as im
import proj1_helpers

## load train.csv return a tuple of 3 arrays:
## 1st one contains the prediction of all elements in order
## 2nd one is an array of arrays containing the data for one prediction
## 3rd is the ids of each prediction
trainResult,trainData,ids = proj1_helpers.load_csv_data('train.csv')

## test.csv return the same expect that testResult is useless (since we want to find it)
testResult,testData,ids = proj1_helpers.load_csv_data('test.csv')





##create dummy result
ids = range(350000,350000+568238)
pred = testResult
#convert the results into the file
proj1_helpers.create_csv_submission(ids,pred,"resultFilePrediction.csv")