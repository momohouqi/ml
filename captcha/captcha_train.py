from tensorflow import keras
from PIL import Image
import glob
import os
import numpy as np
import argparse
import string
import sys

sys.path.append(os.path.dirname(__file__))

#from study import gen_captcha
import gen_captcha

class PNGProcess(object):
    Threshold = 200
    Char_Num = 5
    def __init__(self, train_data_dir):
        self._table = [0 if i < self.Threshold else 1 for i in range(256)]
        self._train_data_dir = train_data_dir

        def _prepare_char_2_int_dict():
            # 0->9: 0->9
            # A->Z: 10->26+10
            nums_str = ''.join([chr(i) for i in range(ord('0'), ord('9') + 1)])
            alpha_str = ''.join([chr(i) for i in range(ord('A'), ord('Z') + 1)])
            whole_char = nums_str + alpha_str
            return {c:i for i, c in enumerate(whole_char)}

        self._char_2_int = _prepare_char_2_int_dict()

    def convert_one_png_file_to_array(self, png_file):
        img = Image.open(png_file)
        img = img.convert('L')
        img = img.point(self._table, '1')
        return np.array(img)

    def convert_string_to_number_char(self, char_str):
        return [self._char_2_int[c] for c in char_str]


    def get_data(self):
        png_files = glob.glob(os.path.join(self._train_data_dir, '*.png'))
        first_one = self.convert_one_png_file_to_array(png_files[0])
        pngs = np.zeros((len(png_files), first_one.shape[0], first_one.shape[1]))
        labels = np.zeros((len(png_files), self.Char_Num))
        for i, png in enumerate(png_files):
            pngs[i] = self.convert_one_png_file_to_array(png)
            labels[i] = self.convert_string_to_number_char(os.path.basename(png).split('.')[0])
        return pngs, labels

def train1(model_file, x_train, y_train, x_test, y_test):
    category_num = 36
    output_char_num = 5
    img_height = 22
    img_width = 63
    channel_num = 1

    input_shape = keras.layers.Input((img_height, img_width, channel_num))
    out = input_shape
    out = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(out)
    #out = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(out)
    out = keras.layers.BatchNormalization()(out)
    out = keras.layers.MaxPooling2D(pool_size=(4, 4), strides=(2,2))(out)
    out = keras.layers.Dropout(0.3)(out)

    out = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(out)
    #out = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(out)
    out = keras.layers.BatchNormalization()(out)
    out = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(out)
    out = keras.layers.Dropout(0.3)(out)

    out = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(out)
    #out = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(out)
    out = keras.layers.BatchNormalization()(out)
    out = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))(out)
    out = keras.layers.Dropout(0.3)(out)

    out = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(out)
    #out = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(out)
    out = keras.layers.BatchNormalization()(out)
    out = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(out)
    # full connect
    out = keras.layers.Flatten()(out)
    out = keras.layers.Dropout(0.3)(out)
    out = [keras.layers.Dense(category_num, name='c{}'.format(i), activation='softmax')(out) for i in range(output_char_num)]

    model = keras.models.Model(inputs=input_shape, outputs=out)

    #model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='Adamax', metrics=['accuracy'])

    #(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, img_height, img_width, channel_num).astype('float32')/255
    y_train = keras.utils.to_categorical(y_train, category_num)
    #y_train = y_train.reshape((y_train.shape[0],  -1))
    y_train = [y_train[:,i,:] for i in range(output_char_num)]

    x_test = x_test.reshape(-1, img_height, img_width, channel_num).astype('float32')/255
    y_test = keras.utils.to_categorical(y_test, category_num)
    #y_test = y_test.reshape((y_test.shape[0], -1))
    y_test = [y_test[:,i,:] for i in range(output_char_num)]

    #model.fit(x_train, y_train, batch_size=100, epochs=3, validation_data=(x_test, y_test))
    history = model.fit(x_train, y_train, batch_size=100, epochs=10)
    #print("history:{}, dir:{}".format(str(history), dir(history)))
    model.evaluate(x=x_test, y=y_test, batch_size=1000)
    if model_file:
        model.save(model_file)
        print("save model to {}".format(model_file))
    return model

class CaptchaTrain(object):
    def __init__(self, model_file, data_file):
        self._model_file = model_file
        self._data_file = data_file

    def train(self):
        train_data_ratio = 0.8
        #data_path = 'output/img_label.npz_compressed.npz'
        data, labels = gen_captcha.load_data(self._data_file)
        print("Load data from {}, their shapes:{}, {}".format(self._data_file, data.shape, labels.shape))
        train_data_size = int(data.shape[0] * train_data_ratio)
        x_train, y_train = data[:train_data_size], labels[:train_data_size]
        x_test, y_test = data[train_data_size:], labels[train_data_size:]
        train1(self._model_file, x_train, y_train, x_test, y_test)
        #print("train data:{}, labels:{}".format(train_data.shape, labels.shape))

    def test(self):
        for png in glob.iglob(os.path.join(self._train_data_dir, '*.png')):
            print("Get png file:{}".format(png))
            self._png_process.convert(png)
            break

def train_model(model_file, data_file):
    ct = CaptchaTrain(model_file, data_file)
    ct.train()

def predict(model_file, imgs_dir):
    img_height = 22
    img_width = 63
    channel_num = 1


    chars = string.digits + string.ascii_uppercase

    model = keras.models.load_model(model_file)
    model.summary()

    pngs = glob.glob(os.path.join(imgs_dir, '*.jpeg'))
    pngs += glob.glob(os.path.join(imgs_dir, '*.png'))
    #imgs_np = np.zeros((len(pngs), img_height, img_width, channel_num))
    for i, img_file in enumerate(pngs):
        img = Image.open(img_file)
        img = img.resize((img_width, img_height))
        img_np = gen_captcha.ImgProcess.convert_one_img_file_to_array(img)
        img_np = img_np.reshape(1, img_height, img_width, 1).astype('float32')/255
        #imgs_np[i] = img_np

        out = model.predict(img_np)
        output = [e.argmax(1)[0] for e in out]
        predict_str = ''.join([chars[char_index] for char_index in output])

        print("img:{}, predict value:{}".format(os.path.basename(img_file), predict_str))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir_to_predict', type=str, default='imgs_from_cnki', help='A folder contains images')
    parser.add_argument('--model_file', type=str, help='trained model file', default='random_space_imgs.model')
    #parser.add_argument('--data_file', type=str, default='output/random_space_imgs.npz', help='train data file')
    parser.add_argument('--data_file', type=str, help='train data file')
    options = parser.parse_args()
    if options.img_dir_to_predict and options.model_file:
        predict(options.model_file, options.img_dir_to_predict)
    elif options.model_file and options.data_file:
        # train
        train_model(options.model_file, options.data_file)

if __name__ == '__main__':
    main()