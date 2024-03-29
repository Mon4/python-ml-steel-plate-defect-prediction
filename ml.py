import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.info()

X_train = train.iloc[:, 1:28].values
Y_train = train.iloc[:, 28:].values
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.3, random_state=42)
X_val = test.iloc[:, 1:28].values
Y_val = test.iloc[:, 28:].values

print(Y_train.shape)
print(Y_test.shape)

def model():
    model = Sequential()
    model.add(Dense(units=27, activation='relu'))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=50, activation='sigmoid'))
    model.add(Dense(units=7))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model


keras_regressor = KerasRegressor(build_fn=create_model, verbose=0)

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=20, batch_size=100)
