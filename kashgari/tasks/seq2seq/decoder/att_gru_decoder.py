# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: gru_att_decoder.py
# time: 9:42 下午

import tensorflow as tf
from kashgari.embeddings.abc_embedding import ABCEmbedding
from kashgari.layers import BahdanauAttention


class AttGRUDecoder(tf.keras.Model):
    def __init__(self,
                 embedding: ABCEmbedding,
                 vocab_size: int,
                 hidden_size: int = 1024):
        super(AttGRUDecoder, self).__init__()
        self.embedding = embedding
        self.gru = tf.keras.layers.GRU(hidden_size,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # 用于注意力
        self.attention = BahdanauAttention(hidden_size)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding.embed_model(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


if __name__ == "__main__":
    pass
