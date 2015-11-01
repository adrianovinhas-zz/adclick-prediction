import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
import scipy as sp

def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = sum(ll)
    ll = ll * -1.0/len(act)
    return ll

train = pd.read_csv("train.csv",nrows=100000)
test = pd.read_csv("test.csv")

y_train = np.ravel(train[['click']])
x_train = train[['banner_pos','device_type','device_conn_type','C1','C14','C15','C16']]#,'C17','C18','C19','C20','C21']]
x_test = test[['banner_pos','device_type','device_conn_type','C1','C14','C15','C16']]#,'C17','C18','C19','C20','C21']]

# x_train.head(10)


enc = OneHotEncoder()
enc.fit(pd.concat([x_train,x_test]))
x_ensembled = enc.transform(pd.concat([x_train,x_test]))

selector = VarianceThreshold(threshold=(.99 * (1 - .99)))
x_reducted = selector.fit_transform(x_ensembled).toarray()

#print(type(x_reducted))
print(x_reducted[0:3,:])
print(x_ensembled.shape)
print(x_reducted.shape)

#x_train = enc.transform(x_train).toarray()
#x_train = x_reducted[:x_train.shape[0],:]
#x_test = x_reducted[x_train.shape[0]:,:]

#print(x_hot_encode.shape)

#x_train = x_hot_encode[:x_hot_encode.shape(0)]
#x_test = x_hot_encode[x_hot_encode.shape(0):]

# print(y_train.shape)

clf = RandomForestRegressor()
clf.fit(x_train,y_train)

y_pred = clf.predict(x_train)


pred = [(y,1-y) for y in y_pred]
act = [(y,(y+1)%2) for y in y_train]



#out = test['id']
#out = pd.DataFrame.from_items([('id', test['id']), ('click', y_pred)])
#print(out.head(10))
print(llfun(act, pred))
#out.to_csv('out.csv',index=False)

