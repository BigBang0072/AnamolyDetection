import sys
import numpy as np
import tensorflow as tf
from keras import optimizers

sys.path.append('/home/abhinav/Desktop/AnamolyDetection/AnamolyDetection/Models')
from RNN_ManyToMany import *
from RNN import *
from RNN_V2 import *



def create_model_single():
    #Generating the Dataset
    m=10
    time_steps=50
    X=np.empty((m,time_steps,1))#dim of imput vector at each is 1-d vector
    Y=np.empty((m,1)) #only one output per sequence
    h_initial=np.empty((m,1)) #only onr inital pre-conditioning per sequence
    for i in range(m):
        h_initial[i,0]=time_steps*(m-1)-1
        last=0
        for j in range(time_steps):
            X[i,j,0]=last=time_steps*(m-1)+j
        Y[i,0]=last+1

    permutation=list(np.random.permutation(m))
    X=X[permutation,:,:]
    Y=Y[permutation,:]
    h_initial=h_initial[permutation,:]

    X_train=X[:700,:,:]
    Y_train=Y[:700,:]
    h_initial_train=h_initial[:700,:]
    X_inputs_train=[h_initial_train,X_train]
    Y_outputs_train=Y_train

    X_test=X[700:-1,:,:]
    Y_test=Y[700:-1,:]
    h_output_test=h_initial[700:-1,:]
    X_inputs_test=[h_output_test,X_test]
    Y_outputs_test=Y_test

    input_shape=(time_steps,1)
    model=get_simpleRNN_model(input_shape,time_steps)
    adam=optimizers.Adam(clipnorm=1.0)
    model.compile(optimizer=adam,loss='mean_absolute_error')
    print(model.summary())

    model.fit(x=X_inputs_train,y=Y_outputs_train,epochs=5000,batch_size=700,validation_data=(X_inputs_test,Y_outputs_test))

def create_model_multi():
    #Generating the Dataset
    m=1000
    time_steps=10
    X=np.empty((m,time_steps,1))#dim of imput vector at each is 1-d vector
    Y=np.empty((m,time_steps,1))#now there is many-to-many relation
    h_initial=np.empty((m,1)) #only onr inital pre-conditioning per sequence
    for i in range(m):
        h_initial[i,0]=time_steps*(m-1)-1
        for j in range(time_steps):
            X[i,j,0]=time_steps*(m-1)+j
            Y[i,j,0]=time_steps*(m-1)+j+1

    permutation=list(np.random.permutation(m))
    X=X[permutation,:,:]
    Y=Y[permutation,:,:]
    h_initial=h_initial[permutation,:]

    X_train=X[:700,:,:]
    Y_train=Y[:700,:,:]
    h_initial_train=h_initial[:700,:]
    X_inputs_train=[h_initial_train,X_train]
    Y_outputs_train=Y_train

    X_test=X[700:-1,:,:]
    Y_test=Y[700:-1,:,:]
    h_output_test=h_initial[700:-1,:]
    X_inputs_test=[h_output_test,X_test]
    Y_outputs_test=Y_test

    input_shape=(time_steps,1)
    model=get_multiOutputRNN_model(input_shape,time_steps)
    #adam=optimizers.Adam(clipnorm=1.0)
    model.compile(optimizer='adam',loss='mean_absolute_error',metrics=['accuracy'])
    print(model.summary())

    model.fit(x=X_inputs_train,y=Y_outputs_train,epochs=5000,batch_size=1000,validation_data=(X_inputs_test,Y_outputs_test)) #,validation_data=(X_inputs_test,Y_outputs_test)

def create_model_trueRNN():
    #Generating the Dataset
    m=1000
    time_steps=10
    X=np.empty((m,time_steps,1))#dim of imput vector at each is 1-d vector
    Y=np.empty((m,time_steps,1))#now there is many-to-many relation

    for i in range(m):
        h_initial[i,0]=time_steps*(m-1)-1
        for j in range(time_steps):
            X[i,j,0]=time_steps*(m-1)+j
            Y[i,j,0]=time_steps*(m-1)+j+1


    hidden_layer_dims=10 #fixed size in all the layer with memory to next time(cuz of this memory)(or we could give it as listOK but later)
    total_mem_layer=1#change simultaneously here and function
    h_initial=np.zeros((m,total_mem_layer,hidden_layer_dims)) #only onr inital pre-conditioning per sequence



    permutation=list(np.random.permutation(m))
    X=X[permutation,:,:]
    Y=Y[permutation,:,:]
    h_initial=h_initial[permutation,:,:]

    X_train=X[:700,:,:]
    Y_train=Y[:700,:,:]
    h_initial_train=h_initial[:700,:,:]
    X_inputs_train=[h_initial_train,X_train]
    Y_outputs_train=Y_train

    X_test=X[700:-1,:,:]
    Y_test=Y[700:-1,:,:]
    h_output_test=h_initial[700:-1,:,:]
    X_inputs_test=[h_output_test,X_test]
    Y_outputs_test=Y_test

    input_shape=(time_steps,1)
    model=get_multiOutputRNN_model(input_shape,time_steps)
    #adam=optimizers.Adam(clipnorm=1.0)
    model.compile(optimizer='adam',loss='mean_absolute_error')
    print(model.summary())

    model.fit(x=X_inputs_train,y=Y_outputs_train,epochs=5000,batch_size=1000,validation_data=(X_inputs_test,Y_outputs_test))

create_model_trueRNN()
