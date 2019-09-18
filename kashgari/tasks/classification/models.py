# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: models.py
# time: 2019-05-22 11:26

import logging
import tensorflow as tf
from typing import Dict, Any
from kashgari.layers import L, AttentionWeightedAverageLayer, KMaxPoolingLayer
from kashgari.tasks.classification.base_model import BaseClassificationModel


class BiLSTM_Model(BaseClassificationModel):

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'layer_bi_lstm': {
                'units': 128,
                'return_sequences': False
            },
            'layer_dense': {
                'activation': 'softmax'
            }
        }

    def build_model_arc(self):
        output_dim = len(self.processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layer_bi_lstm = L.Bidirectional(L.LSTM(**config['layer_bi_lstm']))
        layer_dense = L.Dense(output_dim, **config['layer_dense'])

        tensor = layer_bi_lstm(embed_model.output)
        output_tensor = layer_dense(tensor)

        self.tf_model = tf.keras.Model(embed_model.inputs, output_tensor)


class BiGRU_Model(BaseClassificationModel):

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'layer_bi_gru': {
                'units': 128,
                'return_sequences': False
            },
            'layer_dense': {
                'activation': 'softmax'
            }
        }

    def build_model_arc(self):
        output_dim = len(self.processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layer_bi_gru = L.Bidirectional(L.GRU(**config['layer_bi_gru']))
        layer_dense = L.Dense(output_dim, **config['layer_dense'])

        tensor = layer_bi_gru(embed_model.output)
        output_tensor = layer_dense(tensor)

        self.tf_model = tf.keras.Model(embed_model.inputs, output_tensor)


class CNN_Model(BaseClassificationModel):

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'conv1d_layer': {
                'filters': 128,
                'kernel_size': 5,
                'activation': 'relu'
            },
            'max_pool_layer': {},
            'dense_layer': {
                'units': 64,
                'activation': 'relu'
            },
            'activation_layer': {
                'activation': 'softmax'
            },
        }

    def build_model_arc(self):
        output_dim = len(self.processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        # build model structure in sequent way
        layers_seq = []
        layers_seq.append(L.Conv1D(**config['conv1d_layer']))
        layers_seq.append(L.GlobalMaxPooling1D(**config['max_pool_layer']))
        layers_seq.append(L.Dense(**config['dense_layer']))
        layers_seq.append(L.Dense(output_dim, **config['activation_layer']))

        tensor = embed_model.output
        for layer in layers_seq:
            tensor = layer(tensor)

        self.tf_model = tf.keras.Model(embed_model.inputs, tensor)


class CNN_LSTM_Model(BaseClassificationModel):

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'conv_layer': {
                'filters': 32,
                'kernel_size': 3,
                'padding': 'same',
                'activation': 'relu'
            },
            'max_pool_layer': {
                'pool_size': 2
            },
            'lstm_layer': {
                'units': 100
            },
            'activation_layer': {
                'activation': 'softmax'
            },
        }

    def build_model_arc(self):
        output_dim = len(self.processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layers_seq = []
        layers_seq.append(L.Conv1D(**config['conv_layer']))
        layers_seq.append(L.MaxPooling1D(**config['max_pool_layer']))
        layers_seq.append(L.LSTM(**config['lstm_layer']))
        layers_seq.append(L.Dense(output_dim, **config['activation_layer']))

        tensor = embed_model.output
        for layer in layers_seq:
            tensor = layer(tensor)

        self.tf_model = tf.keras.Model(embed_model.inputs, tensor)


class CNN_GRU_Model(BaseClassificationModel):

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'conv_layer': {
                'filters': 32,
                'kernel_size': 3,
                'padding': 'same',
                'activation': 'relu'
            },
            'max_pool_layer': {
                'pool_size': 2
            },
            'gru_layer': {
                'units': 100
            },
            'activation_layer': {
                'activation': 'softmax'
            },
        }

    def build_model_arc(self):
        output_dim = len(self.processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model
        layers_seq = []
        layers_seq.append(L.Conv1D(**config['conv_layer']))
        layers_seq.append(L.MaxPooling1D(**config['max_pool_layer']))
        layers_seq.append(L.GRU(**config['gru_layer']))
        layers_seq.append(L.Dense(output_dim, **config['activation_layer']))

        tensor = embed_model.output
        for layer in layers_seq:
            tensor = layer(tensor)

        self.tf_model = tf.keras.Model(embed_model.inputs, tensor)


class AVCNN_Model(BaseClassificationModel):

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'spatial_dropout': {
                'rate': 0.25
            },
            'conv_0': {
                'filters': 300,
                'kernel_size': 1,
                'kernel_initializer': 'normal',
                'padding': 'valid',
                'activation': 'relu'
            },
            'conv_1': {
                'filters': 300,
                'kernel_size': 2,
                'kernel_initializer': 'normal',
                'padding': 'valid',
                'activation': 'relu'
            },
            'conv_2': {
                'filters': 300,
                'kernel_size': 3,
                'kernel_initializer': 'normal',
                'padding': 'valid',
                'activation': 'relu'
            },
            'conv_3': {
                'filters': 300,
                'kernel_size': 4,
                'kernel_initializer': 'normal',
                'padding': 'valid',
                'activation': 'relu'
            },
            # ---
            'attn_0': {},
            'avg_0': {},
            'maxpool_0': {},
            # ---
            'maxpool_1': {},
            'attn_1': {},
            'avg_1': {},
            # ---
            'maxpool_2': {},
            'attn_2': {},
            'avg_2': {},
            # ---
            'maxpool_3': {},
            'attn_3': {},
            'avg_3': {},
            # ---
            'v_col3': {
                # 'mode': 'concat',
                'axis': 1
            },
            'merged_tensor': {
                # 'mode': 'concat',
                'axis': 1
            },
            'dropout': {
                'rate': 0.7
            },
            'dense': {
                'units': 144,
                'activation': 'relu'
            },
            'activation_layer': {
                'activation': 'softmax'
            },
        }

    def build_model_arc(self):
        output_dim = len(self.processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layer_embed_dropout = L.SpatialDropout1D(**config['spatial_dropout'])
        layers_conv = [L.Conv1D(**config[f'conv_{i}']) for i in range(4)]
        layers_sensor = []
        layers_sensor.append(L.GlobalMaxPooling1D())
        layers_sensor.append(AttentionWeightedAverageLayer())
        layers_sensor.append(L.GlobalAveragePooling1D())
        layer_view = L.Concatenate(**config['v_col3'])
        layer_allviews = L.Concatenate(**config['merged_tensor'])
        layers_seq = []
        layers_seq.append(L.Dropout(**config['dropout']))
        layers_seq.append(L.Dense(**config['dense']))
        layers_seq.append(L.Dense(output_dim, **config['activation_layer']))

        embed_tensor = layer_embed_dropout(embed_model.output)
        tensors_conv = [layer_conv(embed_tensor) for layer_conv in layers_conv]
        tensors_matrix_sensor = []
        for tensor_conv in tensors_conv:
            tensor_sensors = []
            tensor_sensors = [layer_sensor(tensor_conv) for layer_sensor in layers_sensor]
            # tensor_sensors.append(L.GlobalMaxPooling1D()(tensor_conv))
            # tensor_sensors.append(AttentionWeightedAverageLayer()(tensor_conv))
            # tensor_sensors.append(L.GlobalAveragePooling1D()(tensor_conv))
            tensors_matrix_sensor.append(tensor_sensors)
        tensors_views = [layer_view(list(tensors)) for tensors in zip(*tensors_matrix_sensor)]
        tensor = layer_allviews(tensors_views)
        # tensors_v_cols = [L.concatenate(tensors, **config['v_col3']) for tensors
        #                   in zip(*tensors_matrix_sensor)]
        # tensor = L.concatenate(tensors_v_cols, **config['merged_tensor'])
        for layer in layers_seq:
            tensor = layer(tensor)

        self.tf_model = tf.keras.Model(embed_model.inputs, tensor)


class KMax_CNN_Model(BaseClassificationModel):

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'spatial_dropout': {
                'rate': 0.2
            },
            'conv_0': {
                'filters': 180,
                'kernel_size': 1,
                'kernel_initializer': 'normal',
                'padding': 'valid',
                'activation': 'relu'
            },
            'conv_1': {
                'filters': 180,
                'kernel_size': 2,
                'kernel_initializer': 'normal',
                'padding': 'valid',
                'activation': 'relu'
            },
            'conv_2': {
                'filters': 180,
                'kernel_size': 3,
                'kernel_initializer': 'normal',
                'padding': 'valid',
                'activation': 'relu'
            },
            'conv_3': {
                'filters': 180,
                'kernel_size': 4,
                'kernel_initializer': 'normal',
                'padding': 'valid',
                'activation': 'relu'
            },
            'maxpool_i4': {
                'k': 3
            },
            'merged_tensor': {
                # 'mode': 'concat',
                'axis': 1
            },
            'dropout': {
                'rate': 0.6
            },
            'dense': {
                'units': 144,
                'activation': 'relu'
            },
            'activation_layer': {
                'activation': 'softmax'
            },
        }

    def build_model_arc(self):
        output_dim = len(self.processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layer_embed_dropout = L.SpatialDropout1D(**config['spatial_dropout'])
        layers_conv = [L.Conv1D(**config[f'conv_{i}']) for i in range(4)]
        layers_sensor = [KMaxPoolingLayer(**config['maxpool_i4']),
                         L.Flatten()]
        layer_concat = L.Concatenate(**config['merged_tensor'])
        layers_seq = []
        layers_seq.append(L.Dropout(**config['dropout']))
        layers_seq.append(L.Dense(**config['dense']))
        layers_seq.append(L.Dense(output_dim, **config['activation_layer']))

        embed_tensor = layer_embed_dropout(embed_model.output)
        tensors_conv = [layer_conv(embed_tensor) for layer_conv in layers_conv]
        tensors_sensor = []
        for tensor_conv in tensors_conv:
            tensor_sensor = tensor_conv
            for layer_sensor in layers_sensor:
                tensor_sensor = layer_sensor(tensor_sensor)
            tensors_sensor.append(tensor_sensor)
        tensor = layer_concat(tensors_sensor)
        # tensor = L.concatenate(tensors_sensor, **config['merged_tensor'])
        for layer in layers_seq:
            tensor = layer(tensor)

        self.tf_model = tf.keras.Model(embed_model.inputs, tensor)


class R_CNN_Model(BaseClassificationModel):

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'spatial_dropout': {
                'rate': 0.2
            },
            'rnn_0': {
                'units': 64,
                'return_sequences': True
            },
            'conv_0': {
                'filters': 128,
                'kernel_size': 2,
                'kernel_initializer': 'normal',
                'padding': 'valid',
                'activation': 'relu',
                'strides': 1
            },
            'maxpool': {},
            'attn': {},
            'average': {},
            'concat': {
                'axis': 1
            },
            'dropout': {
                'rate': 0.5
            },
            'dense': {
                'units': 120,
                'activation': 'relu'
            },
            'activation_layer': {
                'activation': 'softmax'
            },
        }

    def build_model_arc(self):
        output_dim = len(self.processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layers_rcnn_seq = []
        layers_rcnn_seq.append(L.SpatialDropout1D(**config['spatial_dropout']))
        layers_rcnn_seq.append(L.Bidirectional(L.GRU(**config['rnn_0'])))
        layers_rcnn_seq.append(L.Conv1D(**config['conv_0']))

        layers_sensor = []
        layers_sensor.append(L.GlobalMaxPooling1D())
        layers_sensor.append(AttentionWeightedAverageLayer())
        layers_sensor.append(L.GlobalAveragePooling1D())
        layer_concat = L.Concatenate(**config['concat'])

        layers_full_connect = []
        layers_full_connect.append(L.Dropout(**config['dropout']))
        layers_full_connect.append(L.Dense(**config['dense']))
        layers_full_connect.append(L.Dense(output_dim, **config['activation_layer']))

        tensor = embed_model.output
        for layer in layers_rcnn_seq:
            tensor = layer(tensor)

        tensors_sensor = [layer(tensor) for layer in layers_sensor]
        tensor_output = layer_concat(tensors_sensor)
        # tensor_output = L.concatenate(tensor_sensors, **config['concat'])

        for layer in layers_full_connect:
            tensor_output = layer(tensor_output)

        self.tf_model = tf.keras.Model(embed_model.inputs, tensor_output)


class AVRNN_Model(BaseClassificationModel):

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'spatial_dropout': {
                'rate': 0.25
            },
            'rnn_0': {
                'units': 60,
                'return_sequences': True
            },
            'rnn_1': {
                'units': 60,
                'return_sequences': True
            },
            'concat_rnn': {
                'axis': 2
            },
            'last': {},
            'maxpool': {},
            'attn': {},
            'average': {},
            'all_views': {
                'axis': 1
            },
            'dropout': {
                'rate': 0.5
            },
            'dense': {
                'units': 144,
                'activation': 'relu'
            },
            'activation_layer': {
                'activation': 'softmax'
            },
        }

    def build_model_arc(self):
        output_dim = len(self.processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layers_rnn0 = []
        layers_rnn0.append(L.SpatialDropout1D(**config['spatial_dropout']))
        layers_rnn0.append(L.Bidirectional(L.GRU(**config['rnn_0'])))

        layer_bi_rnn1 = L.Bidirectional(L.GRU(**config['rnn_1']))

        layer_concat = L.Concatenate(**config['concat_rnn'])

        layers_sensor = []
        layers_sensor.append(L.Lambda(lambda t: t[:, -1], name='last'))
        layers_sensor.append(L.GlobalMaxPooling1D())
        layers_sensor.append(AttentionWeightedAverageLayer())
        layers_sensor.append(L.GlobalAveragePooling1D())

        layer_allviews = L.Concatenate(**config['all_views'])
        layers_full_connect = []
        layers_full_connect.append(L.Dropout(**config['dropout']))
        layers_full_connect.append(L.Dense(**config['dense']))
        layers_full_connect.append(L.Dense(output_dim, **config['activation_layer']))

        tensor_rnn = embed_model.output
        for layer in layers_rnn0:
            tensor_rnn = layer(tensor_rnn)
        tensor_concat = layer_concat([tensor_rnn, layer_bi_rnn1(tensor_rnn)])
        tensor_sensors = [layer(tensor_concat) for layer in layers_sensor]
        tensor_output = layer_allviews(tensor_sensors)
        for layer in layers_full_connect:
            tensor_output = layer(tensor_output)

        self.tf_model = tf.keras.Model(embed_model.inputs, tensor_output)


class Dropout_BiGRU_Model(BaseClassificationModel):

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'spatial_dropout': {
                'rate': 0.15
            },
            'rnn_0': {
                'units': 64,
                'return_sequences': True
            },
            'dropout_rnn': {
                'rate': 0.35
            },
            'rnn_1': {
                'units': 64,
                'return_sequences': True
            },
            'last': {},
            'maxpool': {},
            'average': {},
            'all_views': {
                'axis': 1
            },
            'dropout': {
                'rate': 0.5
            },
            'dense': {
                'units': 72,
                'activation': 'relu'
            },
            'activation_layer': {
                'activation': 'softmax'
            },
        }

    def build_model_arc(self):
        output_dim = len(self.processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layers_rnn = []
        layers_rnn.append(L.SpatialDropout1D(**config['spatial_dropout']))
        layers_rnn.append(L.Bidirectional(L.GRU(**config['rnn_0'])))
        layers_rnn.append(L.Dropout(**config['dropout_rnn']))
        layers_rnn.append(L.Bidirectional(L.GRU(**config['rnn_1'])))

        layers_sensor = []
        layers_sensor.append(L.Lambda(lambda t: t[:, -1], name='last'))
        layers_sensor.append(L.GlobalMaxPooling1D())
        layers_sensor.append(L.GlobalAveragePooling1D())

        layer_allviews = L.Concatenate(**config['all_views'])

        layers_full_connect = []
        layers_full_connect.append(L.Dropout(**config['dropout']))
        layers_full_connect.append(L.Dense(**config['dense']))
        layers_full_connect.append(L.Dense(output_dim, **config['activation_layer']))

        tensor_rnn = embed_model.output
        for layer in layers_rnn:
            tensor_rnn = layer(tensor_rnn)
        tensor_sensors = [layer(tensor_rnn) for layer in layers_sensor]
        tensor_output = layer_allviews(tensor_sensors)
        for layer in layers_full_connect:
            tensor_output = layer(tensor_output)

        self.tf_model = tf.keras.Model(embed_model.inputs, tensor_output)


class Dropout_AVRNN_Model(BaseClassificationModel):

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'spatial_dropout': {
                'rate': 0.25
            },
            'rnn_0': {
                'units': 56,
                'return_sequences': True
            },
            'rnn_dropout': {
                'rate': 0.3
            },
            'rnn_1': {
                'units': 56,
                'return_sequences': True
            },
            'last': {},
            'maxpool': {},
            'attn': {},
            'average': {},
            'all_views': {
                'axis': 1
            },
            'dropout_0': {
                'rate': 0.5
            },
            'dense': {
                'units': 128,
                'activation': 'relu'
            },
            'dropout_1': {
                'rate': 0.25
            },
            'activation_layer': {
                'activation': 'softmax'
            },
        }

    def build_model_arc(self):
        output_dim = len(self.processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layers_rnn = []
        layers_rnn.append(L.SpatialDropout1D(**config['spatial_dropout']))
        layers_rnn.append(L.Bidirectional(L.GRU(**config['rnn_0'])))
        layers_rnn.append(L.SpatialDropout1D(**config['rnn_dropout']))
        layers_rnn.append(L.Bidirectional(L.GRU(**config['rnn_1'])))

        layers_sensor = []
        layers_sensor.append(L.Lambda(lambda t: t[:, -1], name='last'))
        layers_sensor.append(L.GlobalMaxPooling1D())
        layers_sensor.append(AttentionWeightedAverageLayer())
        layers_sensor.append(L.GlobalAveragePooling1D())

        layer_allviews = L.Concatenate(**config['all_views'])
        layers_full_connect = []
        layers_full_connect.append(L.Dropout(**config['dropout_0']))
        layers_full_connect.append(L.Dense(**config['dense']))
        layers_full_connect.append(L.Dropout(**config['dropout_1']))
        layers_full_connect.append(L.Dense(output_dim, **config['activation_layer']))

        tensor_rnn = embed_model.output
        for layer in layers_rnn:
            tensor_rnn = layer(tensor_rnn)
        tensor_sensors = [layer(tensor_rnn) for layer in layers_sensor]
        tensor_output = layer_allviews(tensor_sensors)
        for layer in layers_full_connect:
            tensor_output = layer(tensor_output)

        self.tf_model = tf.keras.Model(embed_model.inputs, tensor_output)


if __name__ == "__main__":
    print(BiLSTM_Model.get_default_hyper_parameters())
    logging.basicConfig(level=logging.DEBUG)
    from kashgari.corpus import SMP2018ECDTCorpus

    x, y = SMP2018ECDTCorpus.load_data()

    import kashgari
    from kashgari.processors.classification_processor import ClassificationProcessor
    from kashgari.embeddings import BareEmbedding

    processor = ClassificationProcessor(multi_label=False)
    embed = BareEmbedding(task=kashgari.CLASSIFICATION, sequence_length=30, processor=processor)
    m = BiLSTM_Model(embed)
    # m.build_model(x, y)
    m.fit(x, y, epochs=2)
    print(m.predict(x[:10]))
    # m.evaluate(x, y)
    print(m.predict_top_k_class(x[:10]))
