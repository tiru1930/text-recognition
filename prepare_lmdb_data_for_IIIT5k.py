import scipy.io
from create_lmdb_dataset import createDataset



class prepareData(object):
	"""docstring for DataLodaer"""
	def __init__(self):
		super(prepareData, self).__init__()
		self.train_data_mat = "./data/IIIT5K/traindata.mat"
		self.test_data_mat = "./data/IIIT5K/testdata.mat"
		self.loaded_train_data_mat = scipy.io.loadmat(self.train_data_mat)
		self.loaded_test_data_mat = scipy.io.loadmat(self.test_data_mat)

	def getImagePathlabel(self):
		testFileName = "./data/TestImagePathLabel.txt"
		trainFileName = "./data/TrainImagePathLabel.txt"
		with open(trainFileName,"w") as tr:
			for trainSample in self.loaded_train_data_mat["traindata"][0]:
				imagePath = trainSample[0][0]
				label 	  = trainSample[1][0]
				tr.write(imagePath+"\t"+label+"\n")	

		with open(testFileName,"w") as te:
			for testSample in self.loaded_test_data_mat["testdata"][0]:
				imagePath = testSample[0][0]
				label 	  = testSample[1][0]
				te.write(imagePath+"\t"+label+"\n")

	def createLmdbDataset(self):
		inputPath = "./data/IIIT5K/"
		gtFile = "./data/TestImagePathLabel.txt"
		outputPath = "./lmdb_data/training/"
		createDataset(inputPath,gtFile,outputPath)
		outputPath = "./lmdb_data/testing/"
		gtFile = "./data/TrainImagePathLabel.txt"
		createDataset(inputPath,gtFile,outputPath)

def main():
	pD = prepareData()
	pD.getImagePathlabel()
	pD.createLmdbDataset()


if __name__ == '__main__':
	main()