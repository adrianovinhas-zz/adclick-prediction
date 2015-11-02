import pandas as pd
import random

class DataHandler:


	def __init__(self,train_file,validation_file):
		self._train = pd.read_csv(train_file)
		self._test = None
		self._ytrain = self._train['click']
		self._ytest = None
		self.train_file = train_file
		self.validation_file = validation_file
		if train_file != validation_file:
			self._test = pd.read_csv(validation_file)
			self._ytest = None
		

	def create_train_and_test(self,sampling):
		if sampling != -1.0:
			self._train = self._train.sample(frac=sampling)
			self._ytrain = self._train['click']

		if self.train_file == self.validation_file:
			rows = random.sample(list(range(0,len(self._train))), int(0.3*len(self._train)))
			self._test = self._train.ix[rows]
			self._train = self._train.drop(rows)		# if no test file has been specified, 30% of the training files will be used for test purposes (70/30 rule)
			self._ytrain = self._train['click']
			self._ytest = self._test['click']


	def transform_data(self,*transfs):
		merged = pd.concat([self._train, self._test],keys=["train","test"])
		counter = 1
		for t in transfs:
			print("Start operation "+str(counter)+" out of "+str(len(transfs))+"...")
			counter += 1
			merged = t(merged)
			print("Finished")

		self._train = merged.ix['train']
		self._test = merged.ix['test']


	def drop(self,features):
		self._train.drop(features,axis=1,inplace=True)
		self._test.drop(features,axis=1,inplace=True)

	def save(self,prefix):
		self._train.to_csv(prefix+"train.csv",index=False)
		self._test.to_csv(prefix+"test.csv",index=False)
