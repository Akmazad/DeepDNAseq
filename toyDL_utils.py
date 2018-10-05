import numpy as np

def load_CSV_data():
	# load 2047 train data (with Label)
	# load 500 test data (with Label)
	
	trainData = np.loadtxt("/short/yr31/aa7970/azData/toy_TrainData.csv", delimiter="\t")
	trainLabel = np.loadtxt("/short/yr31/aa7970/azData/toy_TrainLabel.csv", delimiter="\t")
	testData = np.loadtxt("/short/yr31/aa7970/azData/toy_TestData.csv", delimiter="\t")
	testLabel = np.loadtxt("/short/yr31/aa7970/azData/toy_TestLabel.csv", delimiter="\t")
	
	return (trainData,trainLabel),(testData,testLabel)
	