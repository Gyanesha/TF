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
n = 20000
ds = ds.sample(n)
train = 0.6

#Preparing the dataset
x = ds.iloc[:, [i for i in range(0, 64)]].values    
y = ds.iloc[:, 64].values
x_train, x_test, y_train, y_test = train_test_split( 
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



def create_model(dense_layers=[8],
                 activation='relu',
                 optimizer='rmsprop'):
    model = Sequential()
    # model.add(Dense())
    for index, lsize in enumerate(dense_layers):
        # Input Layer - includes the input_shape
        if index == 0:
            model.add(Dense(lsize,
                            activation=activation,
                            input_dim=64))
        else:
            model.add(Dense(lsize,
                            activation=activation))
            
    model.add(Dense(2,activation='softmax'))
    model.compile(optimizer = optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model,
                        epochs=10, 
                        batch_size=5,
                        verbose=1)

param_grid = {'dense_layers': [[4],[8],[8,8]],
              'activation':['relu','tanh'],
              'optimizer':('rmsprop','adam'),
              'epochs':[10,50],
              'batch_size':[5,16]}

grid = GridSearchCV(model,
                    param_grid=param_grid,
                    return_train_score=True,
                    scoring=['precision_macro','recall_macro','f1_macro'],
                    refit='precision_macro')

grid_results = grid.fit(x_train,y_train)

print('Parameters of the best model: ')
print(grid_results.best_params_)