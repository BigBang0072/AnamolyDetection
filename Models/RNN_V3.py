from keras.layers import Dense,LSTM,SimpleRNN,LSTM,GRU
from keras.layers import BatchNormalization
from keras import optimizers
from keras.models import Sequential

unroll_param=True

def seq_RNN(time_steps):
    model=Sequential()
    model.add(SimpleRNN(20,input_shape=(time_steps,1),activation='relu',return_sequences=True,unroll=unroll_param))
    model.add(SimpleRNN(20,activation='relu',return_sequences=True,unroll=unroll_param))
    model.add(SimpleRNN(1,activation='relu',return_sequences=True,unroll=unroll_param))

    return model

def seq_LSTM(time_steps):
    model=Sequential()
    model.add(LSTM(20,input_shape=(time_steps,1),activation='relu',return_sequences=True,unroll=unroll_param))
    model.add(LSTM(20,activation='relu',return_sequences=True,unroll=unroll_param))
    model.add(LSTM(1,activation='relu',return_sequences=True,unroll=unroll_param))

    return model

def seq_GRU(time_steps):
    model=Sequential()
    model.add(GRU(20,input_shape=(time_steps,1),activation='relu',return_sequences=True,unroll=unroll_param))
    model.add(GRU(20,activation='relu',return_sequences=True,unroll=unroll_param))
    model.add(GRU(1,activation='relu',return_sequences=True,unroll=unroll_param))

    return model
