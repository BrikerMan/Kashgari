# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: gru_decoder.py
# time: 9:41 下午

# type: ignore

import tensorflow as tf

from kashgari.embeddings.abc_embedding import ABCEmbedding


class GRUDecoder(tf.keras.Model):
    def __init__(self,
                 embedding: ABCEmbedding,
                 hidden_size: int,
                 vocab_size: int):
        super(GRUDecoder, self).__init__()
        self.embedding = embedding

        self.gru = tf.keras.layers.GRU(hidden_size,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, dec_input, dec_hidden, enc_output):
        # x 在通过嵌入层后的形状 == （批大小，1，嵌入维度）
        decoder_embedding = self.embedding.embed_model(dec_input)

        s = self.gru(decoder_embedding, initial_state=dec_hidden)
        decoder_outputs, decoder_state = s

        # 输出的形状 == （批大小 * 1，隐藏层大小）
        output = tf.reshape(decoder_outputs, (-1, decoder_outputs.shape[2]))

        # 输出的形状 == （批大小，vocab）
        x = self.fc(output)
        return x, decoder_state, None


if __name__ == "__main__":
    pass
