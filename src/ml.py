from sklearn.ensemble import RandomForestClassifier
import numpy as np

class Model:

	def __init__(self,features):
		self.clf = RandomForestClassifier()
		self._ypred = None
		self.used_features = features

	def train(self,data):
		self.clf.fit(data._train[self.used_features],data._ytrain)

	def log_loss(self,data):
		epsilon = 1e-15
		self._ypred = np.maximum(epsilon, self._ypred)
		self._ypred = np.minimum(1-epsilon, self._ypred)
		ll = sum(data._ytest.ravel()*np.log(self._ypred) + np.subtract(1,data._ytest.ravel())*np.log(np.subtract(1,self._ypred)))
		ll = ll * -1.0/len(self._ypred)
		return ll

	def predict(self,data):
		self._ypred = self.clf.predict_proba(data._test[self.used_features])[:,1]
