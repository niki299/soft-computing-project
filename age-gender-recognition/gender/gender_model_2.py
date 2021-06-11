from keras.callbacks import ModelCheckpoint
from keras_preprocessing.image import ImageDataGenerator

from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense

from gender.gender_data import resize_image, image2gray
from keras.preprocessing.image import img_to_array
import numpy as np


def create_model(width, height, kernels, hidden, classes):
    net = Sequential()

    net.add(Conv2D(kernels, (5, 5), padding="same", input_shape=(height, width, 3)))
    net.add(Activation("relu"))
    net.add(BatchNormalization())

    net.add(Conv2D(kernels, (5, 5), padding="same"))
    net.add(Activation("relu"))
    net.add(BatchNormalization())
    net.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    net.add(Conv2D(kernels, (3, 3), padding="same"))
    net.add(Activation("relu"))
    net.add(BatchNormalization())

    net.add(Conv2D(kernels, (3, 3), padding="same"))
    net.add(Activation("relu"))
    net.add(BatchNormalization())
    net.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    net.add(Flatten())
    net.add(Dense(hidden))
    net.add(Activation("relu"))
    net.add(Dropout(0.5))

    net.add(Dense(hidden))
    net.add(Activation("relu"))

    net.add(Dense(classes))
    net.add(Activation("softmax"))
    return net

def estimate(net, image):

    array = img_to_array(image, data_format="channels_last")

    arrays = []
    arrays.append(array)
    data = np.array(arrays).astype("float") / 255.0

    prob = net.predict(data)
    classes = prob.argmax(axis=1)
    class_num = classes[0]

    return class_num

def predict_gender(img):

    classes = 10
    kernels = 16
    hidden = 256
    imgsize = 128

    img = resize_image(img)
    # img = image2gray(img)

    net = create_model(imgsize, imgsize, kernels, hidden, classes)
    opt = SGD(lr=0.01)
    net.compile(loss="categorical_crossentropy",
                optimizer=opt,
                metrics=["accuracy"])

    net.load_weights('../../gender/weights.best.hdf5')
    net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    gender_group = estimate(net, img)

    gender = ["F", "M"]

    return gender[gender_group]


def training(model, train_path, validation_path):

    training_data_gen = ImageDataGenerator(rescale=1 / 255)
    validation_data_gen = ImageDataGenerator(rescale=1 / 255)

    training_data = training_data_gen.flow_from_directory(
        train_path,
        target_size=(128, 128),
        batch_size=128,
        class_mode='categorical'
    )
    validation_data = validation_data_gen.flow_from_directory(
        validation_path,
        target_size=(128, 128),
        batch_size=128,
        class_mode='categorical'
    )

    filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    history = model.fit(
        training_data,
        epochs=3,
        validation_data=validation_data,
        callbacks=callbacks_list,
        verbose=1
    )

    return model

if __name__ == '__main__':
    train = True
    classes = 10
    kernels = 16
    hidden = 256
    imgsize = 128
    classes = 2

    train_path = "../../../UTKData/UTKFace/Training"
    test_path = "../../../UTKData/UTKFace/Testing"

    net = create_model(imgsize, imgsize, kernels, hidden, classes)
    opt = SGD(lr=0.01)
    net.compile(loss="categorical_crossentropy",
                optimizer=opt,
                metrics=["accuracy"])

    if train:
        net = training(net, train_path, test_path)
    else:
        net.load_weights('weights.best.hdf5')
        net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    netname = str(kernels) + "_" + str(hidden) + ".cnn"
    net.save(netname)


