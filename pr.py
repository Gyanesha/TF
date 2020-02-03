from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow
import os
import tensorflow as tf
import pandas as pd
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from  sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

ds = pd.read_csv("/home/gp/Desktop/Project/dats2.csv")
print(ds.shape)
n = 10000
ds = ds.sample(n)
train = 0.6

#Preparing the dataset
x = ds.iloc[:, [i for i in range(0, 64)]].values    
y = ds.iloc[:, 64].values
xtrain, xtest, ytrain, ytest = train_test_split( 
        x, y, train_size = train, random_state = 200) 

    
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(xtrain, ytrain, epochs=5)

model.evaluate(xtest,  ytest, verbose=2)