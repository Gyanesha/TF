#Dependencies
import numpy as np
import pandas as pd
import os
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score
from  sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#dataset import
ds = pd.read_csv("/home/gp/Desktop/Project/dats2.csv") #You need to change #directory accordingly
ds.head(10) #Return 10 rows of data
from sklearn.model_selection import train_test_split
n = 30000
ds = ds.sample(n)
train = 0.6

#Preparing the dataset
x = ds.iloc[:, [i for i in range(0, 64)]].values    
y = ds.iloc[:, 64].values
xtrain, xtest, ytrain, ytest = train_test_split( 
        x, y, train_size = train, random_state = 200) 

#Changing pandas dataframe to numpy array
# X = dataset.iloc[:,:20].values
# y = dataset.iloc[:,20:21].values

#Normalizing the data
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X = sc.fit_transform(X)

# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder()
# y = ohe.fit_transform(y).toarray()


# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)

#Dependencies
import keras
from keras.models import Sequential
from keras.layers import Dense
# Neural network
# model = Sequential()
# model.add(Dense(16, input_dim=64, activation='relu'))
# model.add(Dense(12, activation='relu'))
# model.add(Dense(1, activation='softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# history = model.fit(xtrain, ytrain, epochs=100, batch_size=64)

# y_pred = model.predict(xtest)
# #Converting predictions to label
# pred = list()
# for i in range(len(y_pred)):
#     pred.append(np.argmax(y_pred[i]))
# #Converting one hot encoded test label to label
# test = list()
# for i in range(len(ytest)):
#     test.append(np.argmax(ytest[i]))

# history = model.fit(xtrain, ytrain,validation_data = (xtest,ytest), epochs=100, batch_size=64)

# number_of_features = 64
# features, target = make_classification(n_samples = 10000,
#                                        n_features = number_of_features,
#                                        n_informative = 3,
#                                        n_redundant = 0,
#                                        n_classes = 2,
#                                        weights = [.5, .5],
#                                        random_state = 0)


def create_model():   
    classifier = Sequential()
    #First Hidden Layer
    classifier.add(Dense(32, activation='relu', kernel_initializer='random_normal', input_dim=64))
    #Second  Hidden Layer
    classifier.add(Dense(32, activation='relu', kernel_initializer='random_normal'))
    #Output Layer
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

    #Compiling the neural network
    classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
    return classifier
# #Fitting the data to the training dataset
# classifier.fit(xtrain,ytrain, batch_size=10, epochs=100)

# eval_model=classifier.evaluate(xtrain, ytrain)
# print(eval_model)
classifier = KerasClassifier(build_fn=create_model, 
                                 epochs=10, 
                                 batch_size=100, 
                                 verbose=0)
predicted = cross_val_score(classifier, x, y, cv=10, n_jobs=8,verbose=1)
print(predicted)
print ("CV Accuracy : %0.8f " % (predicted.mean()))

# y_pred=classifier.predict(xtest)
# y_pred =(y_pred>0.5)

# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(ytest, y_pred)
# print(cm)
# print ("Accuracy : ", accuracy_score(ytest, y_pred))