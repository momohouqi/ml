import tensorflow as tf

import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(cur_dir, '..')))
from third_party.mnist import MnistData
import numpy as np


class MnistTrain(object):
    def __init__(self, img_size, output_size_per_input):
        """img_size,
        output_size_per_input, each image output 10 elements to tell each digit's probability
        """
        self._img_size = img_size
        self._output_size_per_input = output_size_per_input

        self._mnist_data = MnistData(True, True, True)

        """
        1.We want to train a W, each column has 784 weight, and it has 10 this columns, which represents
         pattern for 0, 1,...9;
        2. so for each data input(1, 784), it can get 10 value which represent the probability of 0-9;
            then we get the max of the 10, whose index presents the predict value.
        """
        self._x = tf.placeholder('float', [None, img_size])
        W = tf.Variable(tf.zeros([img_size, output_size_per_input]))
        b = tf.Variable(tf.zeros([output_size_per_input]))

        y = tf.matmul(self._x, W) + b
        """after softmax, each element becomes probability value, so its label should be one hot format
         when getting the loss
        """
        self._y = tf.nn.softmax(y)

        self._sess = tf.InteractiveSession()

    def train(self, iteration_count=1000, batch_size=100):
        train_label = tf.placeholder('float', shape=[None, self._output_size_per_input])

        #self._sess.run(tf.initialize_all_variables())
        self._sess.run(tf.global_variables_initializer())

        cross_entropy = -tf.reduce_sum(train_label*tf.log(self._y))
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

        print('start train...')
        for it in range(iteration_count):
            t_x, t_y = self._mnist_data.get_random_train_data(batch_size)
            self._sess.run(train_step, feed_dict={self._x: t_x, train_label: t_y})
        print('train done')

    def predict(self, x):
        x = x.reshape(-1, 784)
        re = self._sess.run(self._y, feed_dict={self._x: x})
        return re.argmax(1)

    def validate(self):
        test_data, test_label = self._mnist_data.get_test_data()
        def _one_by_one(count):
            for i in range(count):
                predict = self.predict(test_data[i])
                print("predict:{}, lable:{}".format(predict, test_label[i].argmax(0)))
        def _batch(count, show_detail=True):
            xs = test_data[0:count]
            labels = test_label[0:count]
            res = self.predict(xs)
            if show_detail:
                for re, label in zip(res, labels):
                    print("predict:{}, lable:{}".format(re, label.argmax(0)))

            # argmax(1), why 1? 0 for first dimension, the biggest elements, 1 for the second one, biggest in the row
            correction_prediction = tf.equal(res, labels.argmax(1))
            ratio = tf.reduce_mean(tf.cast(correction_prediction, 'float'))
            print('correction ratio:{}'.format(self._sess.run(ratio)))
        _batch(100, False)

def main():
    t = MnistTrain(784, 10)
    t.train()
    t.validate()

if __name__ == '__main__':
    main()
