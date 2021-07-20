import random
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten

from FaceID.file import read_file


class DataSet(object):
    def __init__(self, path):
        self.num_classes = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.img_size = 128
        self.extract_data(path)

    def extract_data(self, path):
        imgs, labels, counter = read_file(path)
        X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.2,
                                                            random_state=random.randint(0, 100))
        X_train = X_train.reshape(X_train.shape[0], 1, self.img_size, self.img_size) / 255.0
        X_test = X_test.reshape(X_test.shape[0], 1, self.img_size, self.img_size) / 255.0
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        Y_train = np_utils.to_categorical(y_train, num_classes=counter)
        Y_test = np_utils.to_categorical(y_test, num_classes=counter)
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.num_classes = counter

    def check(self):
        print('num of dim:', self.X_test.ndim)
        print('shape:', self.X_test.shape)
        print('size:', self.X_test.size)
        print('num of dim:', self.X_train.ndim)
        print('shape:', self.X_train.shape)
        print('size:', self.X_train.size)


class Model(object):
    FILE_PATH = "../face.h5"
    IMAGE_SIZE = 128

    def init(self):
        self.model = None
        self.build_model()

    def read_trainData(self, dataset):
        self.dataset = dataset

    def build_model(self):
        self.model = Sequential()
        self.model.add(
            Convolution2D(
                filters=32,
                kernel_size=(5, 5),
                padding='same',
                # dim_ordering='th',
                input_shape=self.dataset.X_train.shape[1:]
            )
        )
        self.model.add(Activation('relu'))
        self.model.add(
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding="same"
            )
        )
        self.model.add(Convolution2D(filters=64, kernel_size=(5, 5), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        self.model.add(Flatten())
        self.model.add(Dense(1024))
        self.model.add(Activation('relu'))
        self.model.add(Dense(self.dataset.num_classes))
        self.model.add(Activation('softmax'))
        self.model.summary()

    def train_model(self):
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        self.model.fit(self.dataset.X_train, self.dataset.Y_train, epochs=10, batch_size=10)
        print(self.model.summary())

    def evaluate_model(self):
        print('\nTesting----------')
        loss, accuracy = self.model.evaluate(self.dataset.X_test,
                                             self.dataset.Y_test)
        print('test loss;', loss)
        print('test accuracy:', accuracy)

    def save(self, file_path=FILE_PATH):
        print('Model Saved Finished! !!')
        self.model.save(file_path)

    def load(self, file_path=FILE_PATH):
        print('Model Loaded Successful!!!')
        self.model = load_model(file_path)

    def predict(self, img):
        img = img.reshape((1, 1, self.IMAGE_SIZE, self.IMAGE_SIZE))
        img = img.astype('float32')
        img = img / 255.0
        result = self.model.predict(img)
        max_index = np.argmax(result)
        return max_index, result[0][max_index]


if __name__ == '__main__':
    dataset = DataSet('../dataset/')
    model = Model()
    model.read_trainData(dataset)
    model.train_model()
    model.evaluate_model()
    model.save()
