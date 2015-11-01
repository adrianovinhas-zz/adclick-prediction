import pandas as pd
import sys, getopt, os
from preprocessor import DataHandler
from feat_engineering import FeatureEngineering
from ml import Model

class Avazu:

	def __init__(self,sampling,train_file,validation_file,need_feateng):
		self.sampling = sampling 	# sampling = -1 assumes that no sampling is needed and the summarized file already exists
		self.train_file = train_file
		self.validation_file = validation_file
		self.need_feateng = need_feateng

	def main(self):
		
		# Step 1 --> Get and sample data from the original dataset
		print("[AVAZU]\tCreating Data Handler for train and test files...")
		if self.validation_file == None:
			data = DataHandler(self.train_file,self.train_file)
		else:
			data = DataHandler(self.train_file,self.validation_file)

		
 		# you can then get train and test files by doing data._train or data._test
		data.create_train_and_test(self.sampling)
		print("[AVAZU]\tData read and split successfully.")


 		# Step 2 --> Feature Engineering
		feat_eng = FeatureEngineering()
		if need_feateng == True:
			print("[AVAZU]\tStarting feature engineering operations...")
			#data.transform_data(feat_eng.append_hours,feat_eng.append_counters_uniques)
			#data.transform_data(feat_eng.append_days,feat_eng.)
			print("[AVAZU]\tFinished feature engineering operations.")
		
		#print("[AVAZU]\tSaving intermediary dataframe...")
		#data.save("feat_eng_")
		#print("[AVAZU]\tFile saved...")		

		data._train.drop(['click','id'], axis=1, inplace=True)
		data.drop(['C1','device_conn_type','device_type','banner_pos','C15','C16','C17','C18','C19','C20','C21','device_model','site_category','site_domain','site_id','app_category','app_domain','app_id','hours','device_id','device_ip'])
		
		print("[AVAZU]\tSelecting K-Best Features...")
		idxs = feat_eng.select_best_k_features(data._train,data._ytrain,5)
		print(idxs)
		print("[AVAZU]\tFeatures selected.")
 		
 		# Step 3 --> Train model and test it
		print("Training and testing the model...")
		m = Model(idxs)
		m.train(data)
		m.predict(data)

		if self.validation_file != None:
			print(type(data._test['id'].ravel()))
			df = pd.DataFrame({'id':data._test['id'].ravel(),'click':m._ypred}, columns=['id', 'click'])
			df.to_csv('output.csv', index=False)
			print("Output in the file: output.csv")
		else:
			print("Log Loss Result: " + str(m.log_loss(data)))

		def print_help():
			print("python avazu.py <args>\n\n")
			print("------------------------------------------------------------------")
			print("List of arguments:")
			print("\t -t: Specifies a training file to read from a csv file\t[REQUIRED]")
			print("\t -v: Specifies a validation/test file to read from a csv file. If the file is not specified, 30\% of the training samples will be used to build the validation dataset (Default: Not specified)")
			print("\t -s: Specifies a sampling ratio to be performed over the training data. If none is specified, it will use the whole dataset. (Default: Entire dataset)")
			print("\t -f: (To be inproved) Specifies if one desires to apply feature engineering over the dataset or not. (Default: No transformation applied)")
			print("\t -h: Used to print this menu")
			print("------------------------------------------------------------------")



if __name__ == '__main__':
	
	try:
		opts, args = getopt.getopt(sys.argv[1:], "t:s:v:f:h",)
	except getopt.GetoptError as err:
		print(str(err))
		sys.exit(2)

	sampling = -1.0
	train_file = None
	validation_file = None
	need_feateng = False
	for flag, arg in opts:
		if flag == "-s":	# sampling ratio
			sampling = float(arg)
		elif flag == "-t":	# training file
			train_file = arg
		elif flag == "-v":	# validation file
			validation_file = arg
		elif flag == "-f":
			need_feateng = False if arg.lower() == "false" else True
		elif flag =="-h":
			self.print_help()
			sys.exit(0)


	if train_file == None:
		print("No train file defined!")
		sys.exit(2)

	avazu_instance = Avazu(sampling,train_file,validation_file,need_feateng)
	avazu_instance.main()