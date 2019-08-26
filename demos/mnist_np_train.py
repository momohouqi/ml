import numpy as np

import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(cur_dir, '..')))

from third_party.mnist import MnistData

def gradient(loss_fun, W):
    delta = 0.01
    grad = np.zeros_like(W)
    it = np.nditer(W, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp = W[idx]
        W[idx] = tmp + delta
        f1 = loss_fun(W)

        W[idx] = tmp - delta
        f2 = loss_fun(W)

        grad[idx] = (f1 - f2) / (2*delta)
        W[idx] = tmp

        it.iternext()
    return grad

def cross_entropy_error(label, y):
    """
    change first axis of y and label to number of data.
     eg.if y and label are like [0, 0, 1,...0], change it to [[0, 0, 1,...0]]
    """
    if y.ndim == 1:
        label = label.reshape(1, -1)
        y = y.reshape(1, -1)

    batch_size = y.shape[0]
    return -np.sum(label * np.log(y)) / batch_size

def softmax(y):
    return np.exp(y) / np.sum(np.exp(y))

class SimpleNet(object):
    def __init__(self):
        self._mnist_data = MnistData(True, True, True)
        self._W = np.random.rand(784, 10)
        self._b = np.random.rand(10)

    def predict(self, x):
        y = np.dot(x, self._W) + self._b
        return softmax(y)

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(t, y)

    def gradient(self, train_data, train_label):
        loss_func = lambda W: self.loss(train_data, train_label)
        grad_W = gradient(loss_func, self._W)
        grad_b = gradient(loss_func, self._b)
        return grad_W, grad_b

    @staticmethod
    def update_parameter(param, grad, lr):
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            i = it.multi_index
            param[i] -=  lr*grad[i]

            it.iternext()

    def train(self, iteration_count, batch_size = 100, lr=0.1):
        for it in range(iteration_count):
            print("train iteration:{}".format(it))
            train_data, train_label = self._mnist_data.get_random_train_data(batch_size)
            grad_W, grad_b = self.gradient(train_data, train_label)
            self.update_parameter(self._W, grad_W, lr)
            self.update_parameter(self._b, grad_b, lr)
        print("train done")

    def validate(self, validate_count, show_count):
        data, label = self._mnist_data.get_test_data()
        y = self.predict(data[0:validate_count])

        print("labels:{}".format(label[0:show_count].argmax(1)))
        print("predict:{}".format(y[0:show_count].argmax(1)))
        ratio = np.mean(y.argmax(1) == label.argmax(1)[0:validate_count])
        print("ratio:{}".format(ratio))
        return ratio


def main():
    net = SimpleNet()
    ratios = {}
    for i in range(100):
        net.train(10)
        ratio = net.validate(100, 10)
        ratios[(i+1) * 10] = ratio
    print("ratios:{}".format(ratios))
    """
    result:100 iteration:0.57
    result:200 iteration:0.57
    ratios:{
    10: 0.2, 20: 0.36, 30: 0.43, 40: 0.47, 50: 0.57, 60: 0.61, 70: 0.59, 80: 0.62, 90: 0.56, 
    100: 0.65, 110: 0.59, 120: 0.65, 130: 0.68, 140: 0.59, 150: 0.59, 160: 0.59, 170: 0.59, 180: 0.65, 190: 0.66, 
    200: 0.63, 210: 0.6, 220: 0.68, 230: 0.55, 240: 0.63, 250: 0.62, 260: 0.64, 270: 0.62, 280: 0.61, 290: 0.64, 
    300: 0.63, 310: 0.66, 320: 0.63, 330: 0.7, 340: 0.65, 350: 0.71, 360: 0.69, 370: 0.74, 380: 0.75, 390: 0.74, 
    400: 0.72, 410: 0.76, 420: 0.76, 430: 0.79, 440: 0.77, 450: 0.73, 460: 0.79, 470: 0.77, 480: 0.77, 490: 0.77, 
    500: 0.79, 510: 0.75, 520: 0.76, 530: 0.8, 540: 0.78, 550: 0.76, 560: 0.74, 570: 0.75, 580: 0.78, 590: 0.77, 
    600: 0.76, 610: 0.74, 620: 0.75, 630: 0.79, 640: 0.77, 650: 0.73, 660: 0.76, 670: 0.8, 680: 0.75, 690: 0.8, 
    700: 0.8, 710: 0.82, 720: 0.81, 730: 0.78, 740: 0.81, 750: 0.71, 760: 0.83, 770: 0.81, 780: 0.8, 790: 0.8, 
    800: 0.82, 810: 0.82, 820: 0.83, 830: 0.8, 840: 0.81, 850: 0.82, 860: 0.82, 870: 0.84, 880: 0.84, 890: 0.82, 
    900: 0.79, 910: 0.82, 920: 0.83, 930: 0.81, 940: 0.8, 950: 0.81, 960: 0.81, 970: 0.81, 980: 0.84, 990: 0.84, 1000: 0.83}
    """

if __name__ == '__main__':
    main()

