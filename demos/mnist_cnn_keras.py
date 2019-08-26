from tensorflow import keras

def train(model_file):
    model = keras.models.Sequential()

    # first convolution layer
    model.add(keras.layers.Convolution2D(filters=32, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # second convolution layer
    model.add(keras.layers.Convolution2D(filters=64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # full connect
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation='relu'))

    # softmax
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')/255
    y_train = keras.utils.to_categorical(y_train, 10)

    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')/255
    y_test = keras.utils.to_categorical(y_test, 10)

    #model.fit(x_train, y_train, batch_size=200, epochs=1, validation_data=(x_test, y_test))
    history = model.fit(x_train, y_train, batch_size=200, epochs=1)
    print("history:{}, dir:{}".format(str(history), dir(history)))
    model.evaluate(x=x_test, y=y_test, batch_size=100)
    if model_file:
        model.save(model_file)
        print("save model to {}".format(model_file))
    return model

def predict(img_count, model_file):
    model = keras.models.load_model(model_file)
    test_x, test_label = keras.datasets.mnist.load_data()[1]

    import numpy
    marker = numpy.random.choice(test_x.shape[0], img_count)
    test_imgs = test_x[marker]
    test_imgs = test_imgs.reshape(-1, 28, 28, 1) / 255
    res = model.predict(test_imgs, 100)
    predict_values = res.argmax(1)
    labels = test_label[marker]
    print("predict_values:{}".format(predict_values))
    print("labels:{}".format(labels))


if __name__ == '__main__':
    model_file = "cnn_mnist_train_model.h5"
    import os
    if not os.path.exists(model_file):
        train(model_file)
    predict(10, model_file)
