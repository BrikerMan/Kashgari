# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: base_classification_model.py
# time: 2019-05-22 11:23

import random
import logging
import kashgari
from typing import Dict, Any, Tuple, Optional, List
from kashgari.tasks.base_model import BaseModel, BareEmbedding

from kashgari.embeddings.base_embedding import Embedding
from sklearn import metrics


class BaseClassificationModel(BaseModel):

    __task__ = 'classification'

    def __init__(self,
                 embedding: Optional[Embedding] = None,
                 hyper_parameters: Optional[Dict[str, Dict[str, Any]]] = None):
        super(BaseClassificationModel, self).__init__(embedding, hyper_parameters)
        if hyper_parameters is None and \
                self.embedding.processor.__getattribute__('multi_label') is True:
            last_layer_name = list(self.hyper_parameters.keys())[-1]
            self.hyper_parameters[last_layer_name]['activation'] = 'sigmoid'
            logging.warning("Activation Layer's activate function changed to sigmoid for"
                            " multi-label classification question")

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError

    def build_model_arc(self):
        raise NotImplementedError

    def compile_model(self, **kwargs):
        if kwargs.get('loss') is None and self.embedding.processor.multi_label:
            kwargs['loss'] = 'binary_crossentropy'
        super(BaseClassificationModel, self).compile_model(**kwargs)

    def predict(self,
                x_data,
                batch_size=32,
                multi_label_threshold: float = 0.5,
                debug_info=False,
                predict_kwargs: Dict = None):
        """
        Generates output predictions for the input samples.

        Computation is done in batches.

        Args:
            x_data: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
            batch_size: Integer. If unspecified, it will default to 32.
            multi_label_threshold:
            debug_info: Bool, Should print out the logging info.
            predict_kwargs: arguments passed to ``predict()`` function of ``tf.keras.Model``

        Returns:
            array(s) of predictions.
        """
        with kashgari.utils.custom_object_scope():
            tensor = self.embedding.process_x_dataset(x_data)
            pred = self.tf_model.predict(tensor, batch_size=batch_size)
            if self.embedding.processor.multi_label:
                if debug_info:
                    logging.info('raw output: {}'.format(pred))
                pred[pred >= multi_label_threshold] = 1
                pred[pred < multi_label_threshold] = 0

            else:
                pred = pred.argmax(-1)

            res = self.embedding.reverse_numerize_label_sequences(pred)
            if debug_info:
                logging.info('input: {}'.format(tensor))
                logging.info('output: {}'.format(pred))
                logging.info('output argmax: {}'.format(pred.argmax(-1)))
        return res

    def predict_top_k_class(self,
                            x_data,
                            top_k=5,
                            batch_size=32,
                            debug_info=False,
                            predict_kwargs: Dict = None) -> List[Dict]:
        """
        Generates output predictions with confidence for the input samples.

        Computation is done in batches.

        Args:
            x_data: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
            top_k: int
            batch_size: Integer. If unspecified, it will default to 32.
            debug_info: Bool, Should print out the logging info.
            predict_kwargs: arguments passed to ``predict()`` function of ``tf.keras.Model``

        Returns:
            array(s) of predictions.
            single-label classification:
              [
                {
                  "label": "chat",
                  "confidence": 0.5801531,
                  "candidates": [
                    { "label": "cookbook", "confidence": 0.1886314 },
                    { "label": "video", "confidence": 0.13805099 },
                    { "label": "health", "confidence": 0.013852648 },
                    { "label": "translation", "confidence": 0.012913573 }
                  ]
                }
              ]
            multi-label classification:
              [
                {
                  "candidates": [
                    { "confidence": 0.9959336, "label": "toxic" },
                    { "confidence": 0.9358089, "label": "obscene" },
                    { "confidence": 0.6882098, "label": "insult" },
                    { "confidence": 0.13540423, "label": "severe_toxic" },
                    { "confidence": 0.017219543, "label": "identity_hate" }
                  ]
                }
              ]
        """
        if predict_kwargs is None:
            predict_kwargs = {}
        with kashgari.utils.custom_object_scope():
            tensor = self.embedding.process_x_dataset(x_data)
            pred = self.tf_model.predict(tensor, batch_size=batch_size, **predict_kwargs)
            new_results = []

            for sample_prob in pred:
                sample_res = zip(self.label2idx.keys(), sample_prob)
                sample_res = sorted(sample_res, key=lambda k: k[1], reverse=True)
                data = {}
                for label, confidence in sample_res[:top_k]:
                    if 'candidates' not in data:
                        if self.embedding.processor.multi_label:
                            data['candidates'] = []
                        else:
                            data['label'] = label
                            data['confidence'] = confidence
                            data['candidates'] = []
                            continue
                    data['candidates'].append({
                        'label': label,
                        'confidence': confidence
                    })

                new_results.append(data)

            if debug_info:
                logging.info('input: {}'.format(tensor))
                logging.info('output: {}'.format(pred))
                logging.info('output argmax: {}'.format(pred.argmax(-1)))
        return new_results

    def evaluate(self,
                 x_data,
                 y_data,
                 batch_size=None,
                 digits=4,
                 output_dict=False,
                 debug_info=False) -> Optional[Tuple[float, float, Dict]]:
        y_pred = self.predict(x_data, batch_size=batch_size)

        if debug_info:
            for index in random.sample(list(range(len(x_data))), 5):
                logging.debug('------ sample {} ------'.format(index))
                logging.debug('x      : {}'.format(x_data[index]))
                logging.debug('y      : {}'.format(y_data[index]))
                logging.debug('y_pred : {}'.format(y_pred[index]))

        if self.processor.multi_label:
            y_pred_b = self.processor.multi_label_binarizer.fit_transform(y_pred)
            y_true_b = self.processor.multi_label_binarizer.fit_transform(y_data)
            report = metrics.classification_report(y_pred_b,
                                                   y_true_b,
                                                   target_names=self.processor.multi_label_binarizer.classes_,
                                                   output_dict=output_dict,
                                                   digits=digits)
        else:
            report = metrics.classification_report(y_data,
                                                   y_pred,
                                                   output_dict=output_dict,
                                                   digits=digits)
        if not output_dict:
            print(report)
        else:
            return report


if __name__ == "__main__":
    print("Hello world")
