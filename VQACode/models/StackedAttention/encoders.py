import tensorflow as tf

from models.pretrained_models import *


class ImageEncoder(tf.keras.layers.Layer):
    def __init__(self, cnn, d_hidden):
        self.cnn_type = cnn
        super(ImageEncoder, self).__init__()
        self.pretrained_cnn = pretrained_cnn(self.cnn_type, include_top=False)
        self.conv = tf.keras.layers.Conv2D(1024, 3, strides=1, padding='same', activation='relu')
        self.fc = tf.keras.layers.Dense(d_hidden, activation='relu')

    def call(self, img):
        img = cnn_preprocess_input(img, cnn=self.cnn_type)
        img = self.pretrained_cnn(img)
        img = self.conv(img)
        img = tf.keras.layers.Reshape((-1, 1024))(img)
        img = self.fc(img)
        return img


class QuestionEncoder(tf.keras.layers.Layer):
    def __init__(self, d_hidden):
        super(QuestionEncoder, self).__init__()
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=False))
        self.fc = tf.keras.layers.Dense(d_hidden, activation='tanh')

    def call(self, ques):
        feature = self.lstm(ques)
        feature = self.fc(feature)
        return feature
