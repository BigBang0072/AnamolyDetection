import numpy as np
import matplotlib.pyplot as plt

'''
GENERAL NOTE:
A lot of things will be tunable in this accumulation decision
function. Later we will try things which dont require hand-tuned
threshold and other parameters.
Also we could think of using GANs later.
'''



def decide_accumulator(input_time,output_time,model,time_series):
    '''Arguments:
        model       : this will the keras model that we have trained data on.
                        it will be loaded from the saved weight and passed by
                        diff function
        time_seires : this is the full time-series that we are going to apply
                        the analysis on.
      Functionality:
        this function should be able to handle vector inputs i.e multiple
        decision at the same time.
    '''
    shrink_val=1                        #for decrementing counter when no anomlay is there
    increment_val=1                     #for incrementing counter when anomaly is detected
    error_delta=0.0105                   #Tunable
    counter_threshold=1000              #to signal a sustained anomaly after 1 hour
    counter_maximum=int(1.5*counter_threshold)       #large enough for holding sutained anomaly for upto 4 hour
                                        #and small enough for easy decay after anomaly


    #making the threshold decision
    pkt_loss_prediction=predict_packet_loss(input_time,
                                    output_time,time_series,model)
    error=np.abs(pkt_loss_prediction-time_series)                   #should we take jsut the peak or dip too
    #plt.plot(error)
    threshold_decision=error>error_delta
    #plt.plot(threshold_decision*0.8)
    #print(threshold_decision[5000:5500])
    #print(error[5000:5500])
    #print("counter_maximum:",counter_maximum)

    #now we have to design a counter to keep track of anomaly.
    #our data is indexed secondwise so 3600 index/sec makes an hour.
    #so design counter keeping this is ming for them to be large
    #enough for detecting sustained anomaly not just small noise.
    accumulator_decision=np.zeros((time_series.shape[0]))
    counter=0
    for i in range(threshold_decision.shape[0]):
        if(threshold_decision[i]==1):
            counter+=increment_val
        else:
            counter-=shrink_val
        #will have to decide what to do at peak. Read from paper.
        if(counter>=counter_threshold):
            accumulator_decision[i]=1

        #LIMITING CASES.DANGER,we alwyas have to bound the counter.else it will never stop.
        #dont let it go to negetive. Later let it be negetive upto some linit.acc to paper.but bound it
        if(counter>=counter_maximum):
            counter=counter_maximum
        if(counter<=0):
            counter=0 #dont let it go to negetive values

    return pkt_loss_prediction,accumulator_decision


def predict_packet_loss(input_time,output_time,time_series,model):
    '''
    Arguments:
        input_time  : the input duration to the model
        output_time : the output duration of the model
        model       : as defiend above
    Functionality:
        Helper function to make prediction from the model for
        out time_series data for the packet loss. not the decision.
    '''
    total_pairs=int((time_series.shape[0]-input_time)/output_time)  #taking all the possible anteroir as in idea 2.
    predictions=np.zeros((time_series.shape[0]))                    #prediction array

    for i in range(total_pairs):
        fr=output_time*i
        to=output_time*i+(input_time+output_time)
        chunk=time_series[fr:to]
        posterior_data=(chunk[0:input_time]).reshape(1,-1)          #model takes (,input_len)
        anterior_pred=model.predict(posterior_data)                 #predicting on the posterior data
        predictions[fr+input_time:to]=anterior_pred                 #just as elegently as iniitally in dataset creation

    return predictions
