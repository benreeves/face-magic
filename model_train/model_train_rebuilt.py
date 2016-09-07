from __future__ import print_function
import numpy as np
import sequential_model_rebuilt
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils


def load_data():
    """
    Load training data from .npy files.
    """
    X = np.load('rebuilt_112_112_faces.npy')
    y = np.load('rebuilt_labels_encoded.npy')
    X = X.astype(np.float32)
    X /= 255

    seed = np.random.randint(1, 10e5)
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    return X, y


def split_data(X, y, split_ratio=0.15):
    """
    Split data into training and testing.
    :param X: X
    :param y: y
    :param split_ratio: split ratio for train and test data
    """
    split = int(X.shape[0] * split_ratio)
    X_test = X[:split, :, :]
    X_test = X_test.reshape(X_test.shape[0], 1, 112, 112)
    y_test = y[:split, :]
    X_train = X[split:, :, :]
    X_train = X_train.reshape(X_train.shape[0], 1, 112, 112)
    y_train = y[split:, :]
    y_train = y_train[:,0]
    y_test  = y_test[:,0]

    Y_train = np_utils.to_categorical(y_train, 3)
    Y_test = np_utils.to_categorical(y_test, 3)

    return X_train, Y_train, X_test, Y_test


#model.fit_generator(data_generator, samples_per_epoch, nb_epoch)

def create_datagen():

    datagen = ImageDataGenerator(featurewise_center=False,
        samplewise_center=True,
        samplewise_std_normalization=False,
        featurewise_std_normalization=True,
        horizontal_flip=True,
        vertical_flip=False)
    
    datagen.fit(X_train)

    return datagen

if __name__ == "__main__":


    print(50*'-')
    print('loading data')
    X,y = load_data()
    print(50*'-')
    X_train, Y_train, X_test, Y_test = split_data(X, y, split_ratio=0.15)

    print('data succesfully loaded into memory')
    nb_epoch = 50
    batch_size = 32
    samples=X_train.shape[0]

    print(50*'-')
    print('creating image data generator')
    datagen = create_datagen()

    print(50*'-')
    print('building and compiling model')
    model = sequential_model_rebuilt.get_model()

    print(50*'-')
    print('fitting model......')
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
            samples_per_epoch=samples, nb_epoch=nb_epoch,
            validation_data=(X_test, Y_test))
    #model.fit(X_train, Y_train, batch_size = batch_size, nb_epoch=nb_epoch,
	   # validation_data=(X_test, Y_test))
    
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    print('Success!, saving weights')
    model.save_weights('model_weights_gender.hdf5', overwrite=True)
