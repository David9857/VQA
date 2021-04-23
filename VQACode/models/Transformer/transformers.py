import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from models.Transformer.layers import BC_Encoder, Image_Question_Encoder, Knowledge_Question_Encoder, Decoder
from models.pretrained_models import *
from tensorflow.keras.layers import Input, Dense, Reshape, Concatenate, BatchNormalization, Flatten


# class VQATransformer(tf.keras.Model):
#     def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, pe_input, pe_target,
#                  rate=0.1, pretrained_cnn_type="inception", multiplier = 10):
#         super(VQATransformer, self).__init__()
#         self.encoder = Image_Question_Encoder(num_layers, d_model, num_heads, dff,
#                                               pe_input, rate, pretrained_cnn_type, multiplier)
#         self.decoder = Decoder(num_layers, d_model, num_heads, dff, vocab_size, pe_target, rate)
#         self.final_layer = tf.keras.layers.Dense(vocab_size)

#     def call(self, question, img, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
#         enc_output = self.encoder(question, img, training, enc_padding_mask)
#         dec_output, attention_weights = self.decoder(
#             tar, enc_output, training, look_ahead_mask, dec_padding_mask) # (batch, targ_len, d_model)
#         final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
#         return final_output, attention_weights


class VQATransformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, pe_input, pe_target,
                 rate=0.1, pretrained_cnn_type="inception", multiplier = 10):
        super(VQATransformer, self).__init__()
        self.encoder_img = Image_Question_Encoder(num_layers, d_model, num_heads, dff,
                                              pe_input, rate, pretrained_cnn_type, multiplier)
        self.encoder_kn = Knowledge_Question_Encoder(num_layers, d_model, num_heads, dff,
                                              pe_input, rate, pretrained_cnn_type, multiplier)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, vocab_size, pe_target, rate)
        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, question, img, kn, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output_img = self.encoder_img(question, img, training, enc_padding_mask)
        enc_output_kn = self.encoder_kn(question, kn, training, enc_padding_mask)
        
        # Infuse
        # can be replaced by different methods
        enc_output = enc_output_img + enc_output_kn

        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask) # (batch, targ_len, d_model)
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        return final_output, attention_weights

