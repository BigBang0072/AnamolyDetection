import sys
import numpy as np
import tensorflow as tf
from keras import optimizers

sys.path.append('/home/abhinav/Desktop/AnamolyDetection/AnamolyDetection/Models')
from RNN_ManyToMany import *


# # Generating the Dataset
# time_step=3
# m=1000#Number of training example
# X=[]
# Y=np.empty((m,1))
#
# for i in range(time_step+1):
#     if(i==0):
#         h_initial=np.zeros((m,1))
#         X.append(h_initial)
#     else:
#         xt_temp=np.zeros((m,1))
#         for j in range(m):
#             xt_temp[j,0]=time_step*j+(i-1)
#         X.append(xt_temp)
# for i in range(m):
#     Y[i,0]=time_step*(i+1)
#
#
# #print(X[100][:,0]+1==Y[:,0])
# #X_train=
#
# #Creating the model compiling and finally running it.
# tf.set_random_seed(1)
# input_shape=(1,)
# model=get_simpleRNN_model(input_shape,time_step)
# optimizer_handle=optimizer.Adam()
# model.compile(optimizer='adam',loss='mean_absolute_error',metrics=['accuracy'])
# #Dont keep the X and Y as same name as the argument of fit, otherwise unrecognized argument.
# print(model.summary())
# model.fit(X,Y,epochs=5000,batch_size=1000)

def create_model_multi():
    #Generating the Dataset
    m=1000
    time_steps=100
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
    model.compile(optimizer='adam',loss='mean_absolute_error')
    print(model.summary())

    #model.fit(x=X_inputs_train,y=Y_output_train,epochs=5000,batch_size=1000,validation_data=(X_inputs_test,Y_outputs_test))

create_model_multi()
