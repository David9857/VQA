import tensorflow as tf
from models.StackedAttention.encoders import *


class Attention(tf.keras.layers.Layer):
    def __init__(self, d_hidden):
        super(Attention, self).__init__()
        self.d_hidden = d_hidden

    def call(self, img_feature, ques_feature):
        attention_weight = tf.keras.layers.Dot(axes=(2, 1))([img_feature, ques_feature])
        attention_weight = tf.nn.softmax(attention_weight)
        attention_weight = tf.expand_dims(attention_weight, 2)
        img_context = img_feature * attention_weight

        next_feature = tf.reduce_sum(img_context, 1) + ques_feature
        return next_feature


class StackAttention(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_hidden):
        super(StackAttention, self).__init__()
        self.layers = [Attention(d_hidden) for i in range(num_layers)]

    def call(self, img_feature, ques_feature):
        for layer in self.layers:
            ques_feature = layer(img_feature, ques_feature)
        feature = ques_feature
        return feature


class SANet(tf.keras.Model):
    def __init__(self, cnn, num_layers, d_hidden):
        super(SANet, self).__init__()
        self.img_encoder = ImageEncoder(cnn=cnn, d_hidden=d_hidden)
        self.ques_encoder = QuestionEncoder(d_hidden=d_hidden)
        self.stacked_att = StackAttention(num_layers=num_layers, d_hidden=d_hidden)

    def call(self, img, ques):
        img_feature = self.img_encoder(img)
        ques_feature = self.ques_encoder(ques)
        context = self.stacked_att(img_feature, ques_feature)
        return context
