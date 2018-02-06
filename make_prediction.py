import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from simpleFF3001 import *
from plot_decision import *
from make_decision import *
np.random.seed(1)

#adding reuired paths for import or file handlings
sys.path.append('/home/abhinav/Desktop/AnamolyDetection/AnamolyDetection/Models')
time_series_path='time_series3001_5'
filename=time_series_path+'.csv'
metadata_path=time_series_path+'_metadata.npz'

#the link we are going to analyze
link_num=0

#the training input and output dimensions
posterior_min=60    #well give in minutes always. if hour change here only
anterior_min=30     #from here all the works will be converted into seconds index

    ###################### DATASET CREATION ########################
def createDataSet():
    df=pd.read_csv(filename)
    print('Printing Data Sample')
    print(df.head())
    df=df['link'+str(link_num)]
    print('just taking out one of links data:')
    print(df.head())

    time_series=df.as_matrix()
    print("Converting the Series to Array")
    #print(type(time_series))
    print('Shape of time_series: ',time_series.shape)

    #Slicing the Time-series data of a particular link
    posterior_len=60*posterior_min  #giving us of posterior packet loss to condition NN on
    anterior_len=60*anterior_min    #anterior packet loss that net has to predict given the posterior

    #total_pairs=time_series.shape[0]-(posterior_len+anterior_len)+1 #Memory Error
    total_pairs=int((time_series.shape[0]-posterior_len)/anterior_len)
    X=np.empty((total_pairs,posterior_len),dtype=np.float64)
    Y=np.empty((total_pairs,anterior_len),dtype=np.float64)

    chunk_len=posterior_len+anterior_len
    for i in range(total_pairs):
        fr=anterior_len*i               #from where to cut time-series
        to=anterior_len*i+chunk_len     #to where based on current idea2
        chunk=time_series[fr:to]
        X[i,:]=chunk[0:posterior_len]   #we are taking from chunk so start from 0
        Y[i,:]=chunk[posterior_len:]
    print("X_shape: ",X.shape)
    print("Y_shape: ",Y.shape)

    #Splitting the data in train and test split
    perm=np.random.permutation(X.shape[0]) #for shuffling the data (GOOD)
    X=X[perm,:]
    Y=Y[perm,:]
    train_split=int(X.shape[0]*0.75) #75 percent train split.
    X_train=X[0:train_split]
    Y_train=Y[0:train_split]

    X_test=X[train_split:]
    Y_test=Y[train_split:]

    print("Training Input Shape  :",X_train.shape)
    print("Training Output Shape :",Y_train.shape)
    print("Testing Input Shape   :",X_test.shape)
    print("Testing Output Shape  :",Y_test.shape)

    return X_train,Y_train,X_test,Y_test,time_series

def visualizeDataset(X_train,Y_train,X_test,Y_test):
    for i in range(X_train.shape[0]):
        plt.plot(X_train[i,:])
        plt.ylim(0,1)
        plt.xlabel('time_steps')
        plt.ylabel('packet-loss')
    plt.show()
    plt.clf()
    for i in range(Y_train.shape[0]):
        plt.plot(Y_train[i,:])
        plt.ylim(0,1)
        plt.xlabel('time_steps')
        plt.ylabel('packet-loss')
    plt.show()
    plt.clf()
    for i in range(X_test.shape[0]):
        plt.plot(X_test[i,:])
        plt.ylim(0,1)
        plt.xlabel('time_steps')
        plt.ylabel('packet-loss')
    plt.show()
    plt.clf()
    for i in range(Y_test.shape[0]):
        plt.plot(Y_test[i,:])
        plt.ylim(0,1)
        plt.xlabel('time_steps')
        plt.ylabel('packet-loss')
    plt.show()

#createDataSet
X_train,Y_train,X_test,Y_test,time_series=createDataSet()
#visualizeDataset(X_train,Y_train,X_test,Y_test)


    ##################### MODEL CREATION ###########################
    ##################### Prediction Network #######################
# model=simpleFeedForward()
# model.compile(optimizer='adam',loss='mse')
# print(model.summary())
# train_history=model.fit(x=X_train,y=Y_train,epochs=10,
#                             validation_data=(X_test,Y_test))
#prediction=model.predict(X_test) #should see how its doing on train data
# model.save('shifted_anomaly_10_epoch.h5')
    ################## Loading the model for furthur execution######(either this or above one will be commented)
model=load_model('shifted_anomaly_100_epoch.h5')
# train_history=model.fit(x=X_train,y=Y_train,epochs=50,
#                             validation_data=(X_test,Y_test))
# model.save('shifted_anomaly_300_epoch.h5')


    ################## TRAINING VISUALIZATION ######################
input_time=posterior_min*60
output_time=anterior_min*60

#plot_training_losses(train_history)
#plot_predictions(Y_test,prediction)
plot_decision_boundary(link_num,time_series,input_time,output_time,metadata_path,model)
