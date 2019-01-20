# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: cnn_lstm_model.py
@time: 2019-01-19 11:52

"""
from keras.layers import Dense, Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential

from kashgari.tokenizer import Tokenizer
from kashgari.utils import helper
from kashgari.data.corpus import Corpus, SimplifyWeibo4MoodsCorpus


class CNN_LSTM_Model(object):
    def __init__(self):
        self.tokenizer = None
        self.embedding_name = None
        self.model = None

    def _build_model_(self):
        model = Sequential()
        embedding = self._build_embedding_layer_()
        model.add(embedding)
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(100))
        model.add(Dense(len(self.tokenizer.label2idx),
                        activation='sigmoid'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model = model
        self.model.summary()

    def _build_embedding_layer_(self) -> Embedding:
        return Embedding(len(self.tokenizer.word2idx),
                         self.tokenizer.embedding_size,
                         input_length=self.tokenizer.sequence_length,
                         weights=[self.tokenizer.get_embedding_matrix()],
                         trainable=False)

    # def fit(self, x_train, y_train, batch_size: int = 64, epochs: int = 5, **kwargs):
    #     """
    #
    #     :param x_train:
    #     :param y_train:
    #     :param batch_size:
    #     :param epochs:
    #     :param kwargs:
    #     :return:
    #     """
    #     data_path, self.tokenizer = prepare_h5_file(tokenizer=self.tokenizer,
    #                                                 x_data=x_train,
    #                                                 y_data=y_train,
    #                                                 task=k.TaskType.classification)
    #     if not self.model:
    #         self._build_model_()
    #
    #     h5_path = os.path.join(data_path, 'dataset.h5')
    #     h5_file = h5py.File(h5_path, 'r')
    #     data_count = len(h5_file['x'])
    #     train_idx, test_idx = train_test_split(range(data_count), test_size=0.15)
    #
    #     samples = random.sample(list(range(100)), 10)
    #     for index in samples:
    #         x = h5_file['x'][index]
    #         y = h5_file['y'][index]
    #         words = self.tokenizer.token_to_word(x)
    #         labels = self.tokenizer.label_to_token(y)
    #         logging.info('----------- sample {}--------------'.format(index))
    #         logging.info('{} -> {}'.format(words, labels))
    #         logging.info('{} -> {}'.format(x, y))
    #     train_generator = helper.h5f_generator(h5path=h5_path,
    #                                            # indices=train_idx,
    #                                            num_classes=len(self.tokenizer.label2idx),
    #                                            batch_size=batch_size)
    #
    #     test_generator = helper.h5f_generator(h5path=h5_path,
    #                                           # indices=test_idx,
    #                                           num_classes=len(self.tokenizer.label2idx),
    #                                           batch_size=batch_size)
    #
    #     self.model.fit_generator(train_generator,
    #                              steps_per_epoch=len(train_idx) // batch_size,
    #                              epochs=epochs,
    #                              verbose=1,
    #                              callbacks=[],
    #                              validation_data=test_generator,
    #                              validation_steps=len(test_idx) // batch_size)

    def fit_corpus(self,
                   corpus: Corpus,
                   batch_size: int = 128,
                   epochs: int = 10,
                   callbacks=None):
        train_generator = corpus.fit_generator()
        self.tokenizer = corpus.tokenizer
        self._build_model_()
        self.model.fit_generator(train_generator,
                                 steps_per_epoch=corpus.data_count // batch_size,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=callbacks)

    def train(self):
        pass


if __name__ == "__main__":
    from kashgari.utils.logger import init_logger
    import keras
    init_logger()
    tokenizer_conf = SimplifyWeibo4MoodsCorpus.get_recommend_tokenizer_config()
    tokenizer = Tokenizer(**tokenizer_conf)
    corpus = SimplifyWeibo4MoodsCorpus(tokenizer=tokenizer,
                                       data_count_limit=1280)
    classifier = CNN_LSTM_Model()

    save_call_back = keras.callbacks.ModelCheckpoint('./model.model',
                                                     monitor='acc',
                                                     verbose=0,
                                                     save_best_only=True,
                                                     save_weights_only=False,
                                                     mode='auto',
                                                     period=1)
    classifier.fit_corpus(corpus, callbacks=[save_call_back])

