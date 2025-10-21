import keras
from keras import layers

def create_model(input_shape):
    model = keras.Sequential()
    model.add(layers.Dense(10, activation='tanh', input_shape=input_shape))
    model.add(layers.Dense(2, activation='softmax'))  # Output layer for 2 classes (dog, cat)
    
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model