from keras.layers import Input,Dense
from keras.layers import BatchNormalization,Concatenate,Activation
from keras.layers.core import Lambda
from keras import optimizers
from keras.models import Model
import keras.backend as K

def give_timestep_input(x,t):
    return x[:,t,:]
def give_initial_state(h,l):
    return h[:,l,:]
def expand_dims(x):
    return K.expand_dims(x,axis=-1)
def RNN(input_shape,time_steps):
    '''
    Arguments:
        input_shape : the shape of input without the 'm'(size of training data)
        time_steps  : time_steps in the RNN sequence

    Description:
        The weights are shared unlike my mistake in previous version where it
        was not.(that was not actually a RNN, just a big NN)
    '''

        ### ONE-TIME HANDLE and metadata CREATION
    tot_mem_layers=2 #total number of hidden layers that will pass memory
    mem_dim=20
    hidden_layer_dims=[mem_dim,mem_dim]#for now the memory will be passesd from first few layers/later could eb modified to last few
    concatenation_switch=[1,1]#(keep dim same of switch on's)for controlling which hidden layer to have link to next time-step
    output_layer_dim=1
    initializer='glorot_uniform'

    #The (same) concatenation "layer" for concatenation in time at different levels
    concat=Concatenate(axis=-1,name='concat_mem')
    # for i in range(tot_mem_layers):
    #     concat=Concatenate(name='Concat-hidden-'+str(i))
    #     concat_list.append(concat)  #diff for each hidden-layer(cuz they act as node in graph)but same for diff time

    #For normal feed-foreward(same for diff time)
    layer_list=[]
    for i,dim in enumerate(hidden_layer_dims):
        dense=Dense(dim,activation='relu',kernel_initializer=initializer,name='Dense-hidden-'+str(i))
        b_norm=BatchNormalization(name='batch-hidden-'+str(i),axis=-1)
        layer_list.append((dense,b_norm))
    out_layer=Dense(output_layer_dim,activation='relu',kernel_initializer=initializer,name='output')
    out_concat=Concatenate(axis=-1,name='concat_out')



        ###Now UNFOLDING the GRAPH to run RNN through TIME
    X_all=Input(shape=input_shape,name='X_all')
    h_initial=Input(shape=(tot_mem_layers,mem_dim),name='h0')
    X_inputs=[h_initial,X_all]
    prev_mem=[]#for storing previous layer memory layer activation
    next_mem=[]#for storing current layers mem activation for next timestep
    for t in range(time_steps):
        mem_count=0
        next_mem=[]
        for l in range(len(hidden_layer_dims)):
            dense,b_norm=layer_list[l]
            if(l==0):
                X=Lambda(give_timestep_input,arguments={'t':t})(X_all)
                X=dense(X)
                X=b_norm(X)
                if(t==0 and concatenation_switch[mem_count]==1):
                    next_mem.append(X)#keeping the memory for next time step concat
                    h_t=Lambda(give_initial_state,arguments={'l':mem_count})(h_initial)
                    X=concat([h_t,X])
                    mem_count=mem_count+1
                elif(concatenation_switch[mem_count]==1):
                    next_mem.append(X)
                    X=concat([prev_mem[mem_count],X])
                    mem_count=mem_count+1
            else:
                X=dense(X)
                X=b_norm(X)
                if(t==0 and concatenation_switch[mem_count]==1):
                    next_mem.append(X)#keeping the memory for next time step concat
                    h_t=Lambda(give_initial_state,arguments={'l':mem_count})(h_initial)
                    X=concat([h_t,X])
                    mem_count=mem_count+1
                elif(concatenation_switch[mem_count]==1):
                    next_mem.append(X)
                    X=concat([prev_mem[mem_count],X])
                    mem_count=mem_count+1
        Y=out_layer(X)
        if(t==0):
            Y_outputs=Y
        else:
            Y_outputs=out_concat([Y_outputs,Y])
        prev_mem=next_mem

    #Reshaping the accumulated outputs in the required form of output(None,time_steps,1)
    Y_outputs=Lambda(expand_dims)(Y_outputs)

    #Now creating the model
    model=Model(inputs=X_inputs,outputs=Y_outputs,name='RNN-V2')
    return model
