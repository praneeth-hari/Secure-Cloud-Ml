import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import pickle
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
import pickle
from keras.callbacks import ModelCheckpoint
import os
import random

dataset = pd.read_csv("Dataset/test.csv")
labels = np.unique(dataset['Attack_Type']).ravel()

#applying dataset processing technique to convert non-numeric data to numeric data
label_encoder = []
columns = dataset.columns
types = dataset.dtypes.values
for j in range(len(types)):
    name = types[j]
    if name == 'object': #finding column with object type
        le = LabelEncoder()
        dataset[columns[j]] = pd.Series(le.fit_transform(dataset[columns[j]].astype(str)))#encode all str columns to numeric
        label_encoder.append([columns[j], le])
dataset.fillna(0, inplace = True)#replace missing values

#dataset shuffling & Normalization
Y = dataset['Attack_Type'].ravel()
print(np.unique(Y, return_counts=True))
dataset.drop(['Attack_Type'], axis = 1,inplace=True)
X = dataset.values
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)#shuffle dataset values
X = X[indices]
Y = Y[indices]
print("Dataset shuffling & Normalization Completed")

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
y_train1 = to_categorical(y_train)
y_test1 = to_categorical(y_test)

#training ANN algorithm with given hyperparameters
ann_model = Sequential()
#adding ANN dense layer with 32 neurons to filter dataset 64 times
ann_model.add(Dense(32, input_shape=(X_train.shape[1],)))
ann_model.add(Dense(16, activation = 'relu'))
ann_model.add(Dense(y_train1.shape[1], activation = 'softmax'))
ann_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#now train and load the model
if os.path.exists("model/ann_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/ann_weights.hdf5', verbose = 1, save_best_only = True)
    ann_model.fit(X_train, y_train1, batch_size = 32, epochs = 10, validation_data=(X_test, y_test1), callbacks=[model_check_point], verbose=1)
else:
    ann_model.load_weights("model/ann_weights.hdf5")
predict = ann_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test2 = np.argmax(y_test1, axis=1)
acc = accuracy_score(y_test2, predict)
print(acc)

knn = KNeighborsClassifier(algorithm="ball_tree")
knn.fit(X_train[0:1000], y_train[0:1000])
predict = knn.predict(X_test)
acc = accuracy_score(y_test, predict)
print(acc)


best_model = None
best_features = None
ann_model = Sequential()
best_neurons = []

def fitnessFunction(features, train_x, train_y, test_x, test_y, neurons, num_epochs):    
    global ann_model, best_neurons
    best_neurons = neurons
    ann_model.add(Dense(neurons[0], input_shape=(X_train[:, features].shape[1],)))
    ann_model.add(Dense(neurons[1], activation = 'relu'))
    ann_model.add(Dense(y_train1.shape[1], activation = 'softmax'))
    ann_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    ann_model.fit(train_x[:, features], train_y, batch_size = 32, epochs = num_epochs, validation_data=(test_x[:, features], test_y), verbose=1)
    predict = ann_model.predict(test_x[:, features])
    predict = np.argmax(predict, axis=1)
    true_label = np.argmax(test_y, axis=1)
    fitness = accuracy_score(true_label, predict)
    return fitness 


def ABCOptimization(population_size, max_iterations):
    global best_model, best_features
    best = None
    population = [random.sample(range(X_train.shape[1]), int(X_train.shape[1] * 0.8)) for _ in range(population_size)]
    for _ in range(max_iterations):
        fitness_values = [fitnessFunction(features, X_train, y_train1, X_test, y_test1, [64, 32], 4) for features in population]
        for i in range(population_size):
            new_features = population[i].copy()
            new_features[random.randint(0, len(new_features) - 1)] = random.randint(0, X.shape[1] - 1)
            if fitnessFunction(new_features, X_train, y_train1, X_test, y_test1, [64, 32], 10) > fitness_values[i]:
                best_features = new_features
                best_model = ann_model
                best = fitness_values[i]
                break
        if best is not None:
            break

def ANNOptimization(population_size, max_iterations):
    global best_features, best_model, best_neurons
    if os.path.exists("model/abc_ann_weights.h5"):
        data = np.load("model/data.npy", allow_pickle=True)
        best_features = data[0]
        best_neurons = data[1]
        best_epoch = data[2]
        best_lr = data[3]        
        best_model = Sequential()
        best_model.add(Dense(best_neurons[0], input_shape=(X_train[:, best_features].shape[1],)))
        best_model.add(Dense(best_neurons[1], activation = 'relu'))
        best_model.add(Dense(y_train1.shape[1], activation = 'softmax'))
        best_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        best_model.load_weights("model/abc_ann_weights.h5")        
    else:
        ABCOptimization(population_size, max_iterations)
        best_model.save_weights("model/abc_ann_weights.h5")
        data = np.asarray([np.asarray(best_features),np.asarray(best_neurons), 10, 0.001])
        np.save("model/data", data)
        data = np.load("model/data.npy", allow_pickle=True)
        best_features = data[0]
        best_model = load_model("model/abc_ann_weights.h5")
        best_neurons = data[1]
        best_epoch = data[2]
        best_lr = data[3]
    return best_features, best_model, best_neurons, best_epoch, best_lr
        
        
    
            
best_features, best_model, best_neurons, best_epoch, best_lr = ANNOptimization(3, 10)
predict = best_model.predict(X_test[:, best_features])
predict = np.argmax(predict, axis=1)
y_test2 = np.argmax(y_test1, axis=1)
acc = accuracy_score(y_test2, predict)
print(acc)
print(best_features)
print(best_neurons)
print(best_epoch)
print(best_lr)







