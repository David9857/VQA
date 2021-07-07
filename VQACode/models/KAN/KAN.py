import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from models.KAN.layers import Encoder, IEAM, KEAM, Decoder
from models.pretrained_models import *
from tensorflow.keras.layers import Input, Dense, Reshape, Concatenate, BatchNormalization, Flatten


class VQATransformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, pe_input, pe_target, kn_input,
                 rate=0.1, pretrained_cnn_type="inception", multiplier = 10):
        super(VQATransformer, self).__init__()
        self.maximum_position_encoding = pe_target
        self.cam = Encoder(num_layers, d_model, num_heads, dff,
                    pe_input, rate)
        self.ieam = IEAM(num_layers, d_model, num_heads, dff, pe_target, rate, pretrained_cnn_type)
        self.keam = KEAM(num_layers, d_model, num_heads, dff, pe_target, kn_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, vocab_size, pe_target, rate)
        self.img_dense = Dense(d_model, activation='relu')
        self.kn_dense = Dense(d_model, activation='relu')
        self.fc1 = Dense(128 * multiplier)
        self.fc2 = Dense(128 * multiplier)
        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, question, img, kn, tar, training, enc_padding_mask, look_ahead_mask, kn_padding_mask, dec_padding_mask):
        enc_output = self.cam(question, training, enc_padding_mask)
        dec_output1, attention_weights1 = self.ieam(img, enc_output, training, None, dec_padding_mask) # (batch, targ_len, d_model)
        dec_output2, attention_weights2 = self.keam(kn, enc_output, training, kn_padding_mask, dec_padding_mask) # (batch, targ_len, d_model)

        # # img (64,49,512)
        # # img (64,64,512) - inception
        # dec_output1 = self.fc1(dec_output1)  # (64, 49, 350)
        # dec_output1 = tf.transpose(dec_output1, perm=(0, 2, 1))  # (64, 350, 49)
        # dec_output1 = tf.keras.layers.Reshape((128, -1))(dec_output1) # (64, 3, 490)
        # dec_output1 = self.img_dense(dec_output1) # (64, 3, 512)

        # # kn (64,155,512)
        # dec_output2 = self.fc2(dec_output2)  # (64, 155, 350)
        # dec_output2 = tf.transpose(dec_output2, perm=(0, 2, 1))  # (64, 350, 155)
        # dec_output2 = tf.keras.layers.Reshape((128, -1))(dec_output2) # (64, 3, 1550)
        # dec_output2 = self.kn_dense(dec_output2) # (64, 3, 512)

        dec_output = tf.concat([dec_output1, dec_output2], 1)
        dec_output, attention_weights = self.decoder(
            tar, dec_output, training, look_ahead_mask, None) # (batch, targ_len, d_model)
        
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        # attention_weights = (attention_weights1 + attention_weights2)/2
        return final_output, None