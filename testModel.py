import sys
import numpy as np
import tensorflow as tf
from keras import optimizers
import matplotlib.pyplot as plt

sys.path.append('/home/abhinav/Desktop/AnamolyDetection/AnamolyDetection/Models')
from RNN_ManyToMany import *
from RNN import *
from RNN_V2 import *
from RNN_V3 import *

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
    train_split=(m//4)*3
    time_steps=25
    X=np.empty((m,time_steps,1))#dim of imput vector at each is 1-d vector
    Y=np.empty((m,time_steps,1))#now there is many-to-many relation

    for i in range(m):
        for j in range(time_steps):
            X[i,j,0]=time_steps*(i-1)+j
            Y[i,j,0]=time_steps*(i-1)+j+1


    hidden_layer_dims=20 #fixed size in all the layer with memory to next time(cuz of this memory)(or we could give it as listOK but later)
    total_mem_layer=2#change simultaneously here and function
    h_initial=np.zeros((m,total_mem_layer,hidden_layer_dims)) #only onr inital pre-conditioning per sequence

    # #printing the data set for Error check in data-set(if any)
    # print('X_train ',X[:10,:,:])
    # print('Y_train ',Y[:10,:,:])
    # print('h_initial',h_initial[:10,:,:])

    permutation=list(np.random.permutation(m))
    X=X[permutation,:,:]
    Y=Y[permutation,:,:]
    h_initial=h_initial[permutation,:,:]

    X_train=X[:train_split,:,:]
    Y_train=Y[:train_split,:,:]
    h_initial_train=h_initial[:train_split,:,:]
    X_inputs_train=[h_initial_train,X_train]
    Y_outputs_train=Y_train

    X_test=X[train_split:-1,:,:]
    Y_test=Y[train_split:-1,:,:]
    h_initial_test=h_initial[train_split:-1,:,:]
    X_inputs_test=[h_initial_test,X_test]
    Y_outputs_test=Y_test

    # #printing the data set for Error check in data-set(if any)
    # print('X_train ',X[:10,:,:])
    # print('Y_train ',Y[:10,:,:])
    # print('h_initial',h_initial[:10,:,:])

    input_shape=(time_steps,1)
    model=RNN(input_shape,time_steps)
    #adam=optimizers.Adam(clipnorm=1.0)
    model.compile(optimizer='adam',loss='mse')
    print(model.summary())

    train_history=model.fit(x=X_inputs_train,y=Y_outputs_train,epochs=5000,batch_size=m,validation_data=(X_inputs_test,Y_outputs_test))
    plot_training_losses(train_history)

    prediction=model.predict(X_inputs_test)
    plot_predictions(prediction,Y_outputs_test)

def create_model_seqRNN():
    #Generating the Dataset
    m=1000
    train_split=(m//4)*3
    time_steps=25
    X=np.empty((m,time_steps,1))#dim of imput vector at each is 1-d vector
    Y=np.empty((m,time_steps,1))#now there is many-to-many relation

    for i in range(m):
        for j in range(time_steps):
            X[i,j,0]=time_steps*(i-1)+j
            Y[i,j,0]=time_steps*(i-1)+j+1


    permutation=list(np.random.permutation(m))
    X=X[permutation,:,:]
    Y=Y[permutation,:,:]

    X_train=X[:train_split,:,:]
    Y_train=Y[:train_split,:,:]
    Y_outputs_train=Y_train

    X_test=X[train_split:-1,:,:]
    Y_test=Y[train_split:-1,:,:]
    Y_outputs_test=Y_test

    input_shape=(time_steps,1)
    model=seq_LSTM(time_steps)
    #adam=optimizers.Adam(clipnorm=1.0)
    model.compile(optimizer='adam',loss='mse')
    print(model.summary())

    train_history=model.fit(x=X_train,y=Y_train,epochs=5000,batch_size=m,validation_data=(X_test,Y_test))
    plot_training_losses(train_history)

    prediction=model.predict(X_test)
    plot_predictions(prediction,Y_test)

def plot_predictions(pred,actual):
    m=actual.shape[0]
    for i in range(m):
        if(i%10==0):
            plt.plot(pred[i,:,:])
            plt.plot(actual[i,:,:])
            plt.legend(['pred','actual'])
            plt.savefig(str(i)+'.png')
            plt.clf()
def plot_training_losses(train_history):
    loss=train_history.history['loss']
    val_loss=train_history.history['val_loss']

    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['loss','val_loss'])
    plt.show()

create_model_trueRNN()
