import numpy as np
from keras.layers import LSTM,Dense
from keras.models import Model,Sequential

def LSTM_direct():
    model=Sequential()
    model.add(LSTM(1,input_shape=(1,100),return_sequences=True))
    #model.add(Dense(100))
    return model

#creating the Dataset(Train)
m=10
X_train=np.empty((m,1,100))
Y_train=np.empty((m,1,1))
for i in range(m):
    for j in range(100):
        X_train[i,0,j]=100*(i-1)+j
    Y_train[i,0,0]=100*(i-1)+j+1

#creating the test-Dataset
m_test=1
X_test=np.empty((m_test,1,100))
Y_test=np.empty((m_test,1,1))
for i in range(m_test):
    last=0
    for j in range(100):
        X_test[i,0,j]=last=100*i+j+11 #extra offset of 11
    Y_test[i,0,0]=last+1

#getting the model instance
model=LSTM_direct()
model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['accuracy'])
model.fit(x=X_train,y=Y_train,epochs=1000,batch_size=1000,validation_data=(X_test,Y_test))
