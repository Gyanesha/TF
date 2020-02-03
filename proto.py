import os
import numpy as np 
import pandas as pd
# from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from  sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Dropout, Convolution2D, Conv2D, MaxPooling2D, Lambda, GlobalMaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Activation, AveragePooling2D, Concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

os.environ["CUDA_VISIBLE_DEVICES"]="1"


def get_dataset(address):
    ds = pd.read_csv(address)
    return ds

def get_x_y(ds, train): 
    x = ds.iloc[:, [i for i in range(0, 64)]].values
    y = ds.iloc[:, 64].values
    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, train_size = train, random_state = 200)
    return (x, y, xtrain, ytrain, xtest, ytest)   

def create_model(dense_layers=[8], activation='relu', optimizer='rmsprop',init='glorot_uniform'):
    model = Sequential()
    for index, lsize in enumerate(dense_layers):
        if(index == 0):
            model.add(Dense(lsize,kernel_initializer=init, activation=activation,
            input_dim=64))
        else:
            model.add(Dense(lsize, kernel_initializer=init, activation=activation))
    
    model.add(Dense(1,kernel_initializer=init,activation='sigmoid'))
    model.compile(optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    return model


def GridSearch(param_grid,x,y):
    model = KerasClassifier(build_fn=create_model,
                            epochs=10,batch_size=5,
                            verbose=1)

    grid = GridSearchCV(model,
                    param_grid=param_grid,
                    return_train_score=True,
                    scoring=['precision_macro','recall_macro','f1_macro'],
                    refit='precision_macro')

    grid_results = grid.fit(x,y)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    return (grid_results.best_params_, grid_result.best_score_)


def train(n,x,y,xtrain,xtest,ytrain,ytest):
    param_grid = {'dense_layers': [[4],[8],[8,8]],
              'activation':['relu','tanh'],
              'optimizer':('rmsprop','adam'),
              'init' : ['glorot_uniform', 'normal', 'uniform'],
              'epochs':[10,50],
              'batch_size':[5,16]}
    best_params, cv_mean = GridSearch(param_grid, x, y)

    model = create_model(dense_layers=best_params['dense_layers'], activation=best_params['activation'],
            optimizer=best_params['optimizer'],init=best_params['init'])
    history = model.fit(xtrain, ytrain, batch_size=best_params['batch_size'], epochs=best_params['epochs'],
                             verbose=1)
    y_pred = model.predict(xtest)
    # pred = list()
    # for i in range(len(y_pred)):
    #     pred.append(np.argmax(y_pred[i]))
    # #Converting one hot encoded test label to label
    # test = list()
    # for i in range(len(y_test)):
    #     test.append(np.argmax(y_test[i]))
    # model.evaluate(xtest,ytest,verbose =1)
    a = accuracy_score(y_pred,ytest)
    print(a)

    cm = confusion_matrix(ytest, y_pred) 
    
    print ("Confusion Matrix : \n", cm) 
    print ("Accuracy : ", accuracy_score(ytest, y_pred))
    file = open("ann_dats2_keras.txt","a")
    file.write("n="+(str)(n))
    file.write("\n")
    file.write("training: "+(str)(0.6*n))
    file.write("\n")
    file.write("testing: "+(str)(n*(0.4)))
    file.write("\n")
    file.write("Best Parameters:")
    file.write("\n")
    file.write((str)(best_params))
    file.write("\n")
    file.write("CV Accuracy : %0.8f " % (cv_mean))
    file.write("\n")
    # file.write(metrics.classification_report(y, predicted))
    file.write("Confusion Matrix : \n"+(str)(cm))
    file.write("\n")
    file.write("Accuracy : "+(str)(a))
    file.write("\n")
    file.write("---------------------------------------------------------------------------------------------")
    file.write("---------------------------------------------------------------------------------------------")
    file.write("---------------------------------------------------------------------------------------------")
    file.write("                           ")



def main():
    df = get_dataset("dats2.csv")
    arr = [10000,20000,30000,40000,50000,60000,69999]
    for i in arr:
        ds = df.sample(i)
        x,y,xtrain,ytrain,xtest,ytest=get_x_y(ds,0.6)
        train(i,x, y, xtrain,xtest, ytrain, ytest)

if __name__ == '__main__':
    main()