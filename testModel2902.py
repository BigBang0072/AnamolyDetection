import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('/home/abhinav/Desktop/AnamolyDetection/Data/time_series2902.csv')
sys.path.append('/home/abhinav/Desktop/AnamolyDetection/AnamolyDetection/Models')
filename='/home/abhinav/Desktop/AnamolyDetection/Data/time_series2902.csv'
def createDataSet():
    df=pd.read_csv(filename)
    print('Printing Data Sample')
    print(df.head())
    df=df['link0']
    print('just taking out first links data:')
    print(df.head())
    #print(type(df))

    time_series=df.as_matrix()
    print("Converting the Series to Array")
    #print(type(time_series))
    print('Shape of time_series: ',time_series.shape)


    posterior_hour=8
    anterior_hour=1
    posterior_len=60*60*posterior_hour #giving us 8 hours of posterior packet loss to condition NN on
    anterior_len=60*60*posterior_len #one hour anterior packet loss that net has to predict given the posterior

    X=time_series.reshape(-1,posterior_len)
    #print(X.shape)
    ex_size=X.shape[0]

    Y=X[1:ex_size,0:anterior_len]#taking the first hour of every series for prediction of previous series
    X=X[:ex_size-1,:] #dropping the last 8 hout data as it cant be used in training
    print("Shape of X: ",X.shape)
    print("Shape of Y: ",Y.shape)

    X_train=X[:15,:]
    Y_train=Y[:15,:]

    X_test=X[15:,:]
    Y_test=Y[15:,:]

    return X_train,Y_train,X_test,Y_test

X_train,Y_train,X_test,Y_test=createDataSet()
plt.plot(X_test[0,:])
plot
