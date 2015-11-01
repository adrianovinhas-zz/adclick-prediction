
from sklearn.feature_selection import SelectKBest, chi2

class FeatureEngineering:


	# Transform the date string to retrieve only the hours in integers
	def append_hours(self,data):
		hours = data['hour'].ravel()
		for i in range(0,len(hours)):
   			hours[i] = int(str(hours[i])[-2:]) # Fetch only the last two digits (hour)
		data['hours'] = hours
		return data

	# Transform the date string to retrieve only the days in integers
	def append_days(self,data):
		hours = data['hour'].ravel()
		for i in range(0,len(hours)):
   			hours[i] = int(str(hours[i])[-4:-2]) # Fetch only the last two digits (hour)
		data['hours'] = hours
		return data

	# TBD -> User historic info:
	# - ratio of ads clicked before the actual moment, for the the user u
	# - time elapsed starting from the last interaction until the actual interaction moment
	'''def historic(self,data):
		gb = data[['hour','device_id']].groupby('device_id',sort=True)
		for g in gb.groups:'''
			


	# Count each time a value happens within the dataset
	def append_counters_uniques(self,data):    
		features_to_tf = ['C1','device_conn_type','device_type','banner_pos','C15','C16','C17','C18','C19','C20','C21','device_model','site_category','site_domain','site_id','app_category','app_domain','app_id','hours','device_id','device_ip']
		for f in features_to_tf:
			print("\t\t[Count Uniques] -> Feature "+str(f)+" being computed")
			d = {}
			feat_array = data[f].ravel()

			gb = data[['id',f]].groupby(f,sort=False)
			df_counts = gb.count()
	    
			for g in gb.groups:
				d[g] = df_counts.ix[g][0]
	    
			data['counters_'+str(f)] = data[f].apply(lambda x: d[x])
		return data


	#
	# This function is not meant to be used in transform_data!!! It is meant to be used after that phase
	#
	def select_best_k_features(self,data,y,k):
		selector = SelectKBest(chi2, k=k)
		data = selector.fit(data, y)
		idxs = selector.get_support(indices=True)
		return idxs