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

    ageranges = [1, 6, 11, 16, 19, 22, 31, 45, 61, 81, 101]
    classes = len(ageranges) - 1

    train_path = "../../../UTKData/Training"
    test_path = "../../../UTKData/Testing"

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

