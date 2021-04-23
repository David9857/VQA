import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense, Concatenate
from models.pretrained_models import pretrained_cnn
from models.Transformer.multi_head_attention import MultiHeadAttention


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, question, training, mask):
        attn_output, _ = self.mha(question, question, question, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        # print('attn_output', attn_output.shape)
        # print('question', question.shape)
        out1 = self.layernorm1(question + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, kn_padding_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, kn_padding_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.question_dense = tf.keras.layers.Dense(d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, question, training, mask):
        seq_len = tf.shape(question)[1]
        # print('seq_len', seq_len)
        # adding embedding and position encoding.
        x = self.question_dense(question)  # (batch_size, input_seq_len, d_model)
        # print('question shape after dense in encoder', x.shape)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)

class IEAM(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, 
                 maximum_position_encoding, rate=0.1, pretrained_cnn_type='inception'):
        super(IEAM, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        # self.maximum_position_encoding = maximum_position_encoding
        self.pos_encoding = positional_encoding(64, d_model)
        self.pretrained_CNN = pretrained_cnn(pretrained_cnn_type)
        self.conv = tf.keras.layers.Conv2D(1024, 3, padding='same', activation='relu')
        # self.fc1 = Dense(maximum_position_encoding * 10)
        self.image_dense = Dense(d_model, activation='relu')
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)


    def call(self, img, enc_output, training, kn_padding_mask, padding_mask):
        attention_weights = {}

        # shape should be (batchsize, tar_len, _)
        img_feature = self.pretrained_CNN(img)  # (64, 7, 7, 2048)
        img_feature = self.conv(img_feature)   # (64, 7, 7, 1024)
        img_shape = img_feature.shape[-1]     
        img_feature = tf.keras.layers.Reshape((-1, img_shape))(img_feature)  # (64, 49, 1024)
        # img_feature = self.fc1(img_feature)  # (64, 49, 380)
        # img_feature = tf.transpose(img_feature, perm=(0, 2, 1))  # (64, 380, 49)
        # img_feature = tf.keras.layers.Reshape((self.maximum_position_encoding, -1))(img_feature) # (64, 38, 490)
        img_feature = self.image_dense(img_feature)  # (64, 49, 512)
        # 
        img_feature += self.pos_encoding[:, :64, :]
        x = self.dropout(img_feature, training=training)
        # print(x.shape)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, kn_padding_mask, padding_mask)

        attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
        attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        return x, attention_weights    # (batchsize, tar_len, _)

class KEAM(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, 
                 maximum_position_encoding, rate=0.1):
        super(KEAM, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        # self.maximum_position_encoding = maximum_position_encoding
        self.pos_encoding = positional_encoding(194, d_model)
        # self.fc1 = Dense(maximum_position_encoding * 10)
        self.kn_dense = Dense(d_model, activation='relu')

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)


    def call(self, kn, enc_output, training, kn_padding_mask, padding_mask):
        attention_weights = {}
        # (64,155,1024)
        # # shape should be (batchsize, tar_len, _)
        # kn_feature = self.fc1(kn)  # (64, 155, 380)
        # kn_feature = tf.transpose(kn_feature, perm=(0, 2, 1))  # (64, 380, 155)
        # kn_feature = tf.keras.layers.Reshape((self.maximum_position_encoding, -1))(kn_feature) # (64, 38, 1550)
        kn_feature = self.kn_dense(kn)  # (64, 155, 512)
        kn_feature += self.pos_encoding[:, :194, :]
        x = self.dropout(kn_feature, training=training)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, kn_padding_mask, padding_mask)

        attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
        attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        return x, attention_weights    # (batchsize, tar_len, _)
