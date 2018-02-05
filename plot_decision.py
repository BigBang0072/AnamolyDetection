import numpy as np
import matplotlib.pyplot as plt
from make_decision import *

def plot_decision_boundary(link_num,time_series,input_time,output_time,metadata_path,model):
    '''Arguments:
        link_num    : for which we are plotting the analysis
        time_seires : the full time-series of the particular link/or all depend on model
        input_time  : the posterior for predicting the next time-step
        output_time : the anteroir time_stamp being predicted given the posterior
        metadata_path: to be used for getting the metadata created during data creation
        model       : the trained model to make the prediction on
    '''

    #Labelling the GROUND TRUTH (known to us as we generated) by rectangle.
    anomaly_pos,anomaly_min=extractMetadata(metadata_path)
    gt_anomaly_loc=np.zeros((time_series.shape[0]))                 #the ground truth where anomaly is located
    for i in range(anomaly_pos.shape[0]):
        pos=anomaly_pos[i,link_num]
        to=anomaly_min[i,link_num]
        gt_anomaly_loc[pos:pos+to]=0.7
    plt.plot(gt_anomaly_loc[:],label='anomaly_ground_truth')

    #adding the PREDICTION of our model on whole data.
    pkt_loss_predictions,accumulator_decision=decide_accumulator(
                                    input_time,output_time,model,time_series)#take from the decision.py script
    plt.plot(pkt_loss_predictions[:],label='predictions')
    plt.plot(accumulator_decision[:]*0.9,label='accumulator_decision')

    #Adding the ACTUAL TIME-SERIES overlay on the plot.
    plt.plot(time_series[:],label='actual_loss',alpha=0.7) #the case when only one link is there.

    #PLOT-PARAMETERS
    plt.ylim(0,1)
    plt.xlabel('time_steps')
    plt.ylabel('packet_loss')
    plt.legend()
    plt.show()

def extractMetadata(metadata_path):
    metadata=np.load(metadata_path)
    anomaly_pos=metadata['anomaly_pos']
    anomaly_min=metadata['anomaly_min']

    return anomaly_pos,anomaly_min

def plot_training_losses(train_history):
    loss=train_history.history['loss']
    val_loss=train_history.history['val_loss']

    plt.plot(loss)
    plt.plot(val_loss)
    plt.xlabel('epochs')
    plt.ylabel('mean-squared-error')
    plt.legend(['loss','val_loss'])
    plt.show()

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

#add up not down
