import numpy as np
import tensorflow as tf

from keras.layers import Input,Dense,Activation
from keras.layers import Dropout,BatchNormalization
from keras.layers import Concatenate
from keras import optimizers
from keras.models import Model
import keras.backend as K

def multiOutputRNN_oneStep(X_t,ht_prev,time_step):
    '''Arguments:
        X_t     : current time-step input
        ht_prev : activation from the previous time-step
        time-step: for naming the layers. current time-step in sequence

        Description:
         This cell defines the architecture of a single time_step
        unit of an unfolded RNN. This will implement multiple input
        and output unlike its previous version.
        (i.e input and output at each time-step)

        Later:
         Concatenating the memory lanes at different location insted of
         just at the input layer of next time-step.
    '''
    #TUNABLE
    hidden_layer_dims=[10,]  #To be decided
    output_layer_dim=1
    rnn_initializer='glorot_uniform'  #our good ole Xavier init.

    #TUNABLE(could be added directly to hiddenlayer of here ^ see later versions)
    X=Concatenate()([X_t,ht_prev])

    for i,dim in enumerate(hidden_layer_dims):
        X=Dense(dim,activation='relu',kernel_initializer=rnn_initializer,name='time-step '+str(time_step)+'hidden_'+str(i+1))(X)
        X=BatchNormalization(axis=-1)(X)

    #Final Layer of this time-step. here this will be the output of this time along with input
    #to next time-step. Later some or all hidden units coudl be input to next time-step
    X=Dense(output_layer_dim,activation='relu',kernel_initializer='rnn_initializer',name='time-step'+str(time-step)+'output')(X)

    return X #returning the output of this time-step.

def get_multiOutputRNN_model(input_shape,time_steps):
    '''Arguments:
        input_shape : the shape of input (tis time not as list but a Tensor so that we could input all time-step at one during training)
        time_steps  : total time-steps in the RNN sequence

       Description:
        This function will return the many-to-many model of RNN,
        where the input of packet-loos (and other metric) of PerfSonar will
        be taken at each time-step and prediction will be made at each time-step
        about the next packet loss. With this architecture we want to have
        more number of gradient to be updated at each sequrnce, so that
        the problel of vanishing gradient we are encountering in big RNN
        with only one putput per sequrnce could be tackled.
    '''
    #Placeholders Definition (later will be fed, sort of feed dict at fitting time)
        #X_all possible dimesion is (None,time_steps,input_dim of each time-step).None is aoutomatically taken by Keras.
    X_all=Input(shape=input_shape,dtype=tf.float32,name='X_all')
        #the inital pre-conditiong on the sequence.(currently of shape=1 like outputs of each time-step)
    h_initial=Input(shape=(1,),dtype=float32,name='ho')
        #for final inking in the model as input and output.
    X_inputs=[h_initial,X_all]
    Y_outputs=[]

    #Traversing through the time-step to create unfolded version of RNN
    for t in range(time_steps):
        if(t==0):
            Y=multiOutputRNN_oneStep(X_all[t,:,:],h_initial,t+1)
            Y_outputs.append(Y)
        else:
            Y=multiOutputRNN_oneStep(X_all[t,:,:],Y,t+1)
            Y_outputs.append(Y)

    #Now merging all the graph into one model.
    model=Model(inputs=X_inputs,outputs=Y_outputs,name='RNN-V1')

    return model
