from keras.layers import Dense
from keras.models import Sequential


def simpleFeedForward():
    model=Sequential()
    model.add(Dense(2000,input_shape=(3600,),activation='relu'))
    model.add(Dense(1800,activation='relu'))
    model.add(Dense(1800,activation='sigmoid'))

    return model
