# Artificial Neural Network

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Healthdata.csv')
X = dataset.iloc[:, 0:24].values    
y=dataset.iloc[:,24].values               
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)
le1=LabelEncoder()
X[:,0]=le1.fit_transform(X[:,0])
le2=LabelEncoder()
X[:,6]=le2.fit_transform(X[:,6])
le3=LabelEncoder()
X[:,15]=le3.fit_transform(X[:,15])
le4=LabelEncoder()
X[:,16]=le4.fit_transform(X[:,16])
le5=LabelEncoder()
X[:,23]=le5.fit_transform(X[:,23])
y=y.reshape(-1,1)
from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder()
y=onehotencoder.fit_transform(y).toarray()
ohe=OneHotEncoder(categorical_features=[0,6,15,16,23])
X=ohe.fit_transform(X).toarray()
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Building the ANN

# Initializing the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
# Part 3 - Training the ANN
ann =Sequential()
ann.add(Dense(17,input_shape=(34,),activation='relu'))
ann.add(Dense(17,activation='relu'))
ann.add(Dense(10,activation='softmax'))
# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 2, epochs = 100)

# Part 4 - Making the predictions and evaluating the model



# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred = (y_pred==1)
y_test = (y_test==1)
y_pred=onehotencoder.inverse_transform(y_pred)
y_test=onehotencoder.inverse_transform(y_test)

# Predicting the result of a single observation
"""
person
Gender:Male
Weight:65
Age:21
jointime:30 days
avgwaket:7
sleephrs:8
foodtype:junk
type1vt:5
type1vf:3
type2vt:5
type2vf:3
type3vt:5
type3vf:3
type4vt:5
type4vf:3
season:summer
areatype:urban
type1dt:3
type1df:5
type2dt:3
type2df:5
type3dt:3
type3df:5
disease:obesity
"""
new = [['M',65,21,30,7,8,'junk',5,3,5,3,5,3,5,3,'summer','urban',3,5,3,5,3,5,'obesity']]
new=np.array(new)
new[:,0]=le1.transform(new[:,0])
new[:,6]=le2.transform(new[:,6])
new[:,15]=le3.transform(new[:,15])
new[:,16]=le4.transform(new[:,16])
new[:,23]=le5.transform(new[:,23])
new=ohe.transform(new).toarray()
yprednew=ann.predict(sc.transform(new))
yprednew=(yprednew>0.5)
yprednew=onehotencoder.inverse_transform(yprednew)
print(yprednew)

