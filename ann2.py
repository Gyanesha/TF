import os 
import numpy as np
import tensorflow as tf
import pandas as pd
np.random.seed(1337)
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from sklearn.datasets import fetch_mldata
# mnist = fetch_mldata(' /Users/Thomas/Dropbox/Learning/Upwork/tuto_TF/data/mldata/MNIST original')
# print(mnist.data.shape)
# print(mnist.target.shape)

from sklearn.model_selection import train_test_split

print("hello")
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

feature_columns = [tf.feature_column.numeric_column('x', shape=xtrain.shape[1:])]	

estimator = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[300, 100])

print(feature_columns)
# Train the estimator
train_input = tf.estimator.inputs.numpy_input_fn(
    x={"x": xtrain},
    y=ytrain,
    batch_size=50,
    shuffle=False,
    num_epochs=None)
# estimator.train(input_fn = train_input,steps=1000) 
# eval_input = tf.estimator.inputs.numpy_input_fn(
#     x={"x": xtest},
#     y=ytest, 
#     shuffle=False,
#     batch_size=xtest.shape[0],
#     num_epochs=1)
# estimator.evaluate(eval_input,steps=None) 