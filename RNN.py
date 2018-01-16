import numpy as np
import tensorflow as tf

from keras.layers import Input,Dense,Activation
from keras.layers import Dropout,BatchNormalization
from keras.layers import Concatenate
from keras import optimizers
from keras.models import Model
import keras.backend as K


def simpleRNN_oneStep(input_shape,X_t,ht_prev,step_no):
    '''Arguments:
        input_shape: for this time step ie. (None,ToBeDecided)
        ht_prev    : the hidden layer output from the previous time-step
        step_no    : (for naming purpose) The current time-step number
    '''
    #Tunable(AK)
    #the internal normal feed forward dimension to create representation for next time step
    hidden_layer_dims=[10,]    #To be decided
    output_layer_dim=1          #To be decided
    rnn_initializer='glorot_uniform'

    #Making a place holder for current layer input
    X=Concatenate()([X_t,ht_prev]) #Concatenating it to the previous time-step memory

    #Implementing the current layers
    for i,dim in enumerate(hidden_layer_dims):
        X=Dense(dim,activation='relu',kernel_initializer=rnn_initializer,name='time'+str(step_no)+'hidden_'+str(i))(X)
        X=BatchNormalization(axis=-1)(X)

    #Final Layer that will go to next time step.(Currently just passing one layer activation to next step)
    #activation used is 'relu' cuz the packet loss or other metric later will be positive
    X=Dense(output_layer_dim,activation='relu',kernel_initializer=rnn_initializer,name='time'+str(step_no)+'passing')(X)

    return X

def get_simpleRNN_model(input_shape,time_steps):
    '''Arguments:
        input_shape: same as above.
        h_initial  : initial pre-contition i.e the starting state of the system
        time_steps : total time-steps in the recurrent sequence before thhe final output.

        This RNN is of type many-to-one architecture where the
        the input is taken at each time step-discretization of PerfSonar
        but output is generated after certain discrete RNN step(in paper they took
        referrent period of 24 hour time-step.)
    '''
    X_inputs=[]
    h_initial=Input(shape=(1,),dtype=tf.float32,name='h0')
    X_inputs.append(h_initial)
    for i in range(time_steps):
        X_t=Input(input_shape,dtype='float32',name='Input_time'+str(i+1))
        X_inputs.append(X_t)

    #Traversing through the timestep to create the recurrent effect
    for t,X_t in enumerate(X_inputs):
        if(t==1):
            X=simpleRNN_oneStep(input_shape,X_t,X_inputs[0],t)
        elif(t>1):
            X=simpleRNN_oneStep(input_shape,X_t,X,t)

    #This final time-step's NN's output is our prediction of the network parameters based on
    #previous "Referrent Time's  input"

    model=Model(inputs=X_inputs,outputs=X,name='RNN-V0')

    return model

def accuracy_metric(y_true,y_pred):
    return y_pred
