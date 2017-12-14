from data_process import cleanData
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import SGD
import matplotlib.pyplot as plt


def clean():
    global cleanData
    cleanData = np.array(cleanData)

    # spliting data
    X_train = cleanData[:int(0.9 * len(cleanData)), :-1]
    Y_train = cleanData[:int(0.9 * len(cleanData)), -1]

    temp = []
    for i in Y_train:
        temp.append([1, 0]) if i == 2 else temp.append([0, 1])

    Y_train = np.array(temp)

    X_test = cleanData[int(0.9 * len(cleanData)):, :-1]
    Y_test = cleanData[int(0.9 * len(cleanData)):, -1]

    temp = []
    for i in Y_test:
        temp.append([1, 0]) if i == 2 else temp.append([0, 1])

    Y_test = np.array(temp)

    return X_train, Y_train, X_test, Y_test


def train(X_train, Y_train, X_test, Y_test):

    model = Sequential()

    model.add(Dense(units=50, activation='relu', input_dim=4))
    model.add(Dense(units=2, activation='softmax'))

    sgd = SGD(lr=0.001, decay=0.00001)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train, epochs=500,
                        validation_data=(X_test, Y_test))

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = clean()
    train(X_train, Y_train, X_test, Y_test)
