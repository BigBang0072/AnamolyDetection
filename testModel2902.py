import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simpleFF3001 import *
np.random.seed(1)

#adding reuired paths for import or file handlings
sys.path.append('/home/abhinav/Desktop/AnamolyDetection/AnamolyDetection/Models')
time_series_path='time_series3001_5'
filename=time_series_path+'.csv'
metadata_path=time_series_path+'_metadata.npz'

#the link we are going to analyze
link_num=3

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
model=simpleFeedForward()
model.compile(optimizer='adam',loss='mse')
print(model.summary())
train_history=model.fit(x=X_train,y=Y_train,epochs=10,
                            validation_data=(X_test,Y_test))
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

    ################## TRAINING VISUALIZATION ######################
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

    #Labelling the GROUND TRUTH (known to us as we generated) by rectangle.
    anomaly_pos,anomaly_min=extractMetadata(metadata_path)
    gt_anomaly_loc=np.zeros((time_series.shape[0]))                 #the ground truth where anomaly is located
    for i in range(anomaly_pos.shape[0]):
        pos=anomaly_pos[i,link_num]
        to=anomaly_min[i,link_num]
        gt_anomaly_loc[pos:pos+to]=0.9
    plt.plot(gt_anomaly_loc[:],label='anomaly_ground_truth')

    #adding the PREDICTION of our model on whole data.
    total_pairs=int((time_series.shape[0]-input_time)/output_time)  #taking all the possible anteroir as in idea 2.
    predictions=np.zeros((time_series.shape[0]))                    #prediction array
    
    for i in range(total_pairs):
        fr=output_time*i
        to=output_time*i+(input_time+output_time)
        chunk=time_series[fr:to]
        posterior_data=(chunk[0:input_time]).reshape(1,-1)          #model takes (,input_len)
        anterior_pred=model.predict(posterior_data)                 #predicting on the posterior data
        predictions[fr+input_time:to]=anterior_pred                 #just as elegently as iniitally in dataset creation
    plt.plot(predictions[:],label='predictions')

    #Adding the ACTUAL TIME-SERIES overlay on the plot.
    plt.plot(time_series[:],label='actual_loss',alpha=0.7) #the case when only one link is there.

    #PLOT-PARAMETERS
    plt.ylim(0,1)
    plt.xlabel('time_steps')
    plt.ylabel('packet_loss')
    plt.legend()
    plt.show()


input_time=posterior_min*60
output_time=anterior_min*60
plot_decision_boundary(link_num,time_series,input_time,output_time,model)
