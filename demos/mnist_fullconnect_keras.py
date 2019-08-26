import tensorflow as tf  # 深度学习库，Tensor 就是多维数组

def train(epoch_num):
    # build a model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # prepare data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    # train
    model.fit(x_train, y_train, epochs=epoch_num)

    # evaluate
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print("loss:{}".format(val_loss))
    print("accuracy:{}".format(val_acc))

if __name__ == '__main__':
    train(1)
