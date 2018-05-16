import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    for index, _ in enumerate(series):
        if index + window_size < len(series):
            X.append(series[index:index+window_size])
            y.append(series[index+window_size])
        
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    
    # chars = sorted(list(set(text)))
    # [' ', '"', '$', '%', '&', "'", '(', ')', '*', '-', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '@', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'à', 'â', 'è', 'é']
    lowercase_characters = [chr(i) for i in range(97, 97 + 26)]
    
    cleaned_text = ''
    
    for character in text:
        if character in punctuation or character in lowercase_characters:
            cleaned_text = cleaned_text + character
        else:
            cleaned_text = cleaned_text + ' '
            
    return cleaned_text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    for index in range(0, len(text) - window_size, step_size):
        inputs.append(text[index:index+window_size])
        outputs.append(text[index+window_size])        
        
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    pass
