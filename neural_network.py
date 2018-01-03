from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras.utils import  np_utils
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from data import loadImages

def predict(model):
    print("\n>>>PREDICTING ON TEST DATA<<<")
    # testSimpleX,testSimpleY = loadImages("images/TEST_SIMPLE");
    print("\n\tLoading images...")
    testX,testY = loadImages("processed_images/TEST");
    print("\tDONE! Images are loaded.")

    encoder = LabelEncoder()
    encoder.fit(testY)
    encoded_y_test = encoder.transform(testY)
    testY = np_utils.to_categorical(encoded_y_test)

    prediction_matching =  np.rint(model.predict(testX))

    print(accuracy_score(testY, prediction_matching))


def fitModel(epochs, batch_size):
    print("\n>>>TRAINING MODEL<<<")
    print("\n\tLoading images...")
    trainX, trainY = loadImages("processed_images/TRAIN")
    print("\tDONE! Images are loaded.")
    encoder = LabelEncoder()
    encoder.fit(trainY)
    encoded_y_train = encoder.transform(trainY)
    trainY = np_utils.to_categorical(encoded_y_train)

    model = neural_network_model()
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size)
    model.save_weights('network.h5')
    return model

def loadModel(path):
    model = neural_network_model()
    model.load_weights(path)
    return model

def neural_network_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(240, 320, 3,), output_shape=(240, 320, 3,)))

    model.add(Conv2D(32, (3, 3), input_shape=(240, 320, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.7))
    model.add(Dense(4))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


