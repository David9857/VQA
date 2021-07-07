import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from models.Transformer.layers import BC_Encoder, Image_Question_Encoder, Knowledge_Question_Encoder, Decoder, CoAttEncoder
from models.pretrained_models import *
from tensorflow.keras.layers import Input, Dense, Reshape, Concatenate, BatchNormalization, Flatten



class VQATransformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, pe_input, pe_target,
                 rate=0.1, pretrained_cnn_type="inception", multiplier = 10):
        super(VQATransformer, self).__init__()
        self.pe_input = pe_input
        self.encoder_img = Image_Question_Encoder(num_layers, d_model, num_heads, dff,
                            pe_input, rate, pretrained_cnn_type, multiplier)
        # self.encoder_img = CoAttEncoder(num_layers, d_model, num_heads, dff, pe_input, pretrained_cnn_type)
        self.encoder_kn = Knowledge_Question_Encoder(num_layers, d_model, num_heads, dff,
                           pe_input, rate, pretrained_cnn_type, multiplier)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, vocab_size, pe_target, rate)
        # self.fc1 = Dense(pe_input * multiplier)
        # self.dense_kn = tf.keras.layers.Dense(d_model)
        # self.dense = tf.keras.layers.Dense(d_model)
        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, question, img, kn, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        # enc_output_qus, enc_output_img = self.encoder_img(question, img, training, enc_padding_mask)
        enc_output_img, attn_weights = self.encoder_img(question, img, training, enc_padding_mask)
        enc_output_kn = self.encoder_kn(question, kn, training, enc_padding_mask)
        # kn = self.fc1(kn)  # (64, 147, 380)
        # kn = tf.transpose(kn, perm=(0, 2, 1))  # (64, 380, 147)
        # kn = tf.keras.layers.Reshape((self.pe_input, -1))(kn) # (64, 38, 1470)
        # kn = self.dense_kn(kn)  # (64, 38, 512)
        # Infuse
        # can be replaced by different methods
        enc_output = enc_output_img + enc_output_kn
        # enc_output = enc_output_img
        # enc_output = tf.concat([enc_output_img, kn],1)
        # enc_output = self.dense(enc_output)

        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask) # (batch, targ_len, d_model)
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        return final_output, [attn_weights,attention_weights]

