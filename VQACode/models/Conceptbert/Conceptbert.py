import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from models.Conceptbert.layers import Encoder, CoAttEncoder, Decoder
from models.pretrained_models import *
from tensorflow.keras.layers import Input, Dense, Reshape, Concatenate, BatchNormalization, Flatten
from models.Conceptbert.cti import CTI

class VQATransformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, pe_input, pe_target, rank, dz,
                 rate=0.1, pretrained_cnn_type="inception", multiplier = 10):
        super(VQATransformer, self).__init__()
        self.nq = pe_input
        self.dz = dz
        self.ca_encoder = CoAttEncoder(num_layers, d_model, num_heads, dff, pe_input, pretrained_cnn_type) # (num_layers, d_model, num_heads, dff, maximum_position_encoding)                           
        self.kg_encoder = Encoder(num_layers, d_model, num_heads, dff, pe_input)
        # self.cti = CTI(rank, d_model, d_model, d_model, dz)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, vocab_size, pe_target, rate)
        self.kg_dense = tf.keras.layers.Dense(d_model)
        self.cvl_dense = tf.keras.layers.Dense(pe_input*dz)
        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, question, img, kg, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        question, img = self.ca_encoder(question, img, training, enc_padding_mask)   # padding mask = None
        kg = self.kg_dense(kg)
        kg = self.kg_encoder(question, kg, training, enc_padding_mask)
        # (b, dz)
        # cvl_embd = self.cti(img, question, kg)
        # # (b, 1, dz)
        # cvl_embd = tf.expand_dims(cvl_embd, -2)
        # # (b, 1, nq*dz)
        # cvl_embd = self.cvl_dense(cvl_embd)
        # # (b, nq, dz)
        # cvl_embd = tf.reshape(cvl_embd, [-1, self.nq, self.dz])
        cvl_embd = (question + img)/2
        cvl_embd = tf.concat([cvl_embd, kg],1)
        dec_output, attention_weights = self.decoder(
            tar, cvl_embd, training, look_ahead_mask, None) # (batch, targ_len, d_model)
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        return final_output, attention_weights


