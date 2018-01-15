import numpy as np
import tensorflow as tf

from keras.layers import Input,Dense,Activation
from keras.layers import Dropout,BatchNormalization
from keras.layers import Concatenate
from keras import optimizers
from keras.models import Model


# Generating the Dataset
time_step=3
m=1000#Number of training example
X=[]
Y=np.empty((m,1))

for i in range(time_step+1):
    if(i==0):
        h_initial=np.zeros((m,1))
        X.append(h_initial)
    else:
        xt_temp=np.zeros((m,1))
        for j in range(m):
            xt_temp[j,0]=time_step*j+(i-1)
        X.append(xt_temp)
for i in range(m):
    Y[i,0]=time_step*(i+1)


#print(X[100][:,0]+1==Y[:,0])
#X_train=

#Creating the model compiling and finally running it.
tf.set_random_seed(1)
input_shape=(1,)
model=get_simpleRNN_model(input_shape,time_step)
optimizer_handle=optimizer.Adam()
model.compile(optimizer='adam',loss='mean_absolute_error',metrics=['accuracy'])
#Dont keep the X and Y as same name as the argument of fit, otherwise unrecognized argument.
print(model.summary())
model.fit(X,Y,epochs=5000,batch_size=1000)
