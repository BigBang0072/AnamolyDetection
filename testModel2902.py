import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simpleFF3001 import *

sys.path.append('/home/abhinav/Desktop/AnamolyDetection/AnamolyDetection/Models')
time_series_path='/home/abhinav/Desktop/AnamolyDetection/Data/time_series3001_10'
filename=time_series_path+'.csv'
metadata_path=time_series_path+'_metadata.npz'
link_num=3
posterior_min=60
anterior_min=30

def createDataSet():
    df=pd.read_csv(filename)
    print('Printing Data Sample')
    print(df.head())
    df=df['link'+str(link_num)]
    print('just taking out first links data:')
    print(df.head())
    #print(type(df))

    time_series=df.as_matrix()
    print("Converting the Series to Array")
    #print(type(time_series))
    print('Shape of time_series: ',time_series.shape)

    posterior_len=60*posterior_min #giving us 1 hours of posterior packet loss to condition NN on
    anterior_len=60*anterior_min #30 minutes anterior packet loss that net has to predict given the posterior

    X=time_series.reshape(-1,posterior_len)
    #print(X.shape)
    ex_size=X.shape[0]

    Y=X[1:ex_size,0:anterior_len]#taking the first hour of every series for prediction of previous series
    X=X[:ex_size-1,:] #dropping the last 8 hout data as it cant be used in training
    print("Shape of X: ",X.shape)
    print("Shape of Y: ",Y.shape)

    X_train=X[:125,:]
    Y_train=Y[:125,:]

    X_test=X[125:,:]
    Y_test=Y[125:,:]

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

#createDataSet()
X_train,Y_train,X_test,Y_test,time_series=createDataSet()
#visualizeDataset(X_train,Y_train,X_test,Y_test)

model=simpleFeedForward()
model.compile(optimizer='adam',loss='mse')
print(model.summary())
train_history=model.fit(x=X_train,y=Y_train,epochs=15,validation_data=(X_test,Y_test))
prediction=model.predict(X_test) #should see how its doing on train data

def plot_predictions(actual,pred):
    m=actual.shape[0]
    for i in range(m):
    #if(i%10==0):
        plt.plot(pred[i,:])
        plt.plot(actual[i,:],alpha=0.5)
        plt.legend(['pred','actual'])
        plt.ylim(0,1)
        plt.xlabel('time_steps')
        plt.ylabel('packet-loss')
        plt.savefig(str(i)+'.png')
        plt.clf()
def plot_training_losses(train_history):
    loss=train_history.history['loss']
    val_loss=train_history.history['val_loss']

    plt.plot(loss)
    plt.plot(val_loss)
    plt.xlabel('epochs')
    plt.ylabel('mean-squared-error')
    plt.legend(['loss','val_loss'])
    plt.show()

plot_training_losses(train_history)
#plot_predictions(Y_test,prediction)

def extractMetadata(metadata_path):
    metadata=np.load(metadata_path)
    anomaly_pos=metadata['anomaly_pos']
    anomaly_min=metadata['anomaly_min']

    return anomaly_pos,anomaly_min

def plot_decision_boundary(link_num,time_series,input_time,output_time,model):
    '''Arguments:
        link_num    : for which we are plotting the analysis
        time_seires : the full time-series of the particular link/or all depend on model
        input_time  : the posterior for predicting the next time-step
        output_time : the anteroir time_stamp being predicted given the posterior
        model       : the trained model to make the prediction on
    '''

    anomaly_pos,anomaly_min=extractMetadata(metadata_path)
    gt_anomaly_loc=np.zeros((time_series.shape[0])) #the ground truth where anomaly is located

    for i in range(anomaly_pos.shape[0]):
        pos=anomaly_pos[i,link_num]
        to=anomaly_min[i,link_num]
        gt_anomaly_loc[pos:pos+to]=0.9
    plt.plot(gt_anomaly_loc[:],label='anomaly_ground_truth')

    total_blocks=time_series.shape[0]/output_time    #total number of blocks of size input shape
                                                    #also we would have to leave last block
    #we cant access last block cuz then there will be no actiul value to compre with in output time
    predictions=np.zeros((time_series.shape[0]))
    print(total_blocks-2)#have to change here later
    for i in range(int(total_blocks-2)):
        posterior_data=time_series[i*output_time:i*output_time+input_time]
        posterior_data=posterior_data.reshape(1,-1)
        #print(posterior_data.shape)
        anterior_pred=model.predict(posterior_data)
        #will have to chage here later
        predictions[(i+2)*output_time:(i+3)*output_time]=anterior_pred

    plt.plot(predictions[:],label='predictions')
    plt.plot(time_series[:],label='actual_loss',alpha=0.7) #the case when only one link is there.
    plt.ylim(0,1)
    plt.xlabel('time_steps')
    plt.ylabel('packet_loss')
    plt.legend()
    plt.show()


input_time=posterior_min*60
output_time=anterior_min*60
plot_decision_boundary(link_num,time_series,input_time,output_time,model)
