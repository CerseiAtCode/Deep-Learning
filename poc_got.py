import flask
import numpy as npy
import tensorflow as tf
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.models import load_model
# load GOT characters death dataset
import pandas as pds
path='C:/Users/rhalde/Documents/blog/blog/character-predictions.csv'

Data_A = pds.read_csv(path, usecols= [7, 16, 17, 18, 19, 20, 25, 26, 28, 29, 30, 31])
Data_B = pds.read_csv(path, usecols=[32])

print (Data_A)
print (Data_B)

# Splitting the dataset into the Training set and Test set
train_dataA,test_dataA,train_dataB,test_dataB = train_test_split(Data_A.values, Data_B.values, test_size = 0.3)

print (train_dataA.shape)
print (test_dataA.shape)
print (train_dataB.shape)
print (test_dataB.shape)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_dataA = sc.fit_transform(train_dataA)
test_dataA = sc.transform(test_dataA)

# create model
from keras.models import Sequential
from keras.layers import Dense
neural_model = Sequential()
neural_model.add(Dense(15, input_dim=12, activation='relu'))
neural_model.add(Dense(15, activation='relu'))
neural_model.add(Dense(15, activation='relu'))
neural_model.add(Dense(1, activation='sigmoid'))
neural_model.summary()

# Compile model
neural_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# log for tensorboard graph purpose
import keras
tbCallBack = keras.callbacks.TensorBoard(log_dir='/tmp/keras_logs', write_graph=True)

# Fit the model
neural_model.fit(train_dataA, train_dataB, epochs=200, batch_size=50,  verbose=1, callbacks=[tbCallBack])

# Predicting the Test set results
Y_pred = neural_model.predict(test_dataA)
Y_pred = (Y_pred > 0.6)


# Calculating Model Accuracy
from sklearn.metrics import accuracy_score
acs = accuracy_score(test_dataB, Y_pred)
print("\nAccuracy Score: %.2f%%" % (acs * 100))


from keras.models import model_from_json


"""# Save the model
neural_model.save('gotCharactersDeathPredictions.h5')
del neural_model"""


import pickle

pkl_filename = "got_model.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(neural_model, file)

# Load from file
with open(pkl_filename, 'rb') as file:  
    pickle_model = pickle.load(file)
    print("read file")
    
    
    






