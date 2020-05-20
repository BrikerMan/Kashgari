# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: labeling_report.py
# time: 7:00 下午

"""
Seqeval Code from: https://github.com/chakki-works/seqeval

Added directly to the project to solve install issue.

Metrics to assess performance on sequence labeling task given prediction
Functions named as ``*_score`` return a scalar value to maximize: the higher
the better
"""

from collections import defaultdict
from typing import List, Dict, Tuple, Any

import numpy as np


def bulk_get_entities(seq_list: List[List[str]], *, suffix: bool = False) -> List[Tuple[str, int, int]]:
    seq = [item for sublist in seq_list for item in sublist + ['O']]
    return get_entities(seq, suffix=suffix)


def get_entities(seq: List[str], *, suffix: bool = False) -> List[Tuple[str, int, int]]:
    """Gets entities from sequence.
    Args:
        seq: sequence of labels.
        suffix:
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        >>> from kashgari.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        if suffix:
            tag = chunk[-1]
            type_ = chunk.split('-')[0]
        else:
            tag = chunk[0]
            type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag: str, tag: str, prev_type: str, type_: str) -> bool:
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E':
        chunk_end = True
    if prev_tag == 'S':
        chunk_end = True

    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'S':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag: str, tag: str, prev_type: str, type_: str) -> bool:
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def f1_score(y_true: List[List[str]],
             y_pred: List[List[str]],
             suffix: bool = False) -> float:
    """Compute the F1 score.
    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::
        F1 = 2 * (precision * recall) / (precision + recall)
    Args:
        y_true: 2d array. Ground truth (correct) target values.
        y_pred: 2d array. Estimated targets as returned by a tagger.
        suffix:
    Returns:
        score : float.
    Example:
        >>> from kashgari.metrics.sequence_labeling import f1_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> f1_score(y_true, y_pred)
        0.50
    """
    true_entities = set(bulk_get_entities(y_true, suffix=suffix))
    pred_entities = set(bulk_get_entities(y_pred, suffix=suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return score


def accuracy_score(y_true: List[List[str]], y_pred: List[List[str]]) -> float:
    """Accuracy classification score.
    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        >>> from kashgari.metrics.sequence_labeling import accuracy_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> accuracy_score(y_true, y_pred)
        0.80
    """
    y_true_all = [item for sublist in y_true for item in sublist]
    y_pred_all = [item for sublist in y_pred for item in sublist]

    nb_correct = sum(y_t == y_p for y_t, y_p in zip(y_true_all, y_pred_all))
    nb_true = len(y_true)

    score = nb_correct / nb_true

    return score


def precision_score(y_true: List[List[str]],
                    y_pred: List[List[str]],
                    suffix: bool = False) -> float:
    """Compute the precision.
    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample.
    The best value is 1 and the worst value is 0.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        >>> from kashgari.metrics.sequence_labeling import precision_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> precision_score(y_true, y_pred)
        0.50
    """
    true_entities = set(bulk_get_entities(y_true, suffix=suffix))
    pred_entities = set(bulk_get_entities(y_pred, suffix=suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)

    score = nb_correct / nb_pred if nb_pred > 0 else 0

    return score


def recall_score(y_true: List[List[str]],
                 y_pred: List[List[str]],
                 suffix: bool = False) -> float:
    """Compute the recall.
    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.
    The best value is 1 and the worst value is 0.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        >>> from kashgari.metrics.sequence_labeling import recall_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> recall_score(y_true, y_pred)
        0.50
    """
    true_entities = set(bulk_get_entities(y_true, suffix=suffix))
    pred_entities = set(bulk_get_entities(y_pred, suffix=suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_true = len(true_entities)

    score = nb_correct / nb_true if nb_true > 0 else 0

    return score


def performance_measure(y_true: List[List[str]],
                        y_pred: List[List[str]]) -> Dict[str, int]:
    """
    Compute the performance metrics: TP, FP, FN, TN
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        performance_dict : dict
    Example:
        >>> from kashgari.metrics.sequence_labeling import performance_measure
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'O', 'B-ORG'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> performance_measure(y_true, y_pred)
        (3, 3, 1, 4)
    """
    performance_dict = dict()
    y_true_all = [item for sublist in y_true for item in sublist]
    y_pred_all = [item for sublist in y_pred for item in sublist]

    performance_dict['TP'] = sum(y_t == y_p for y_t, y_p in zip(y_true_all, y_pred_all)
                                 if ((y_t != 'O') or (y_p != 'O')))
    performance_dict['FP'] = sum(y_t != y_p for y_t, y_p in zip(y_true_all, y_pred_all))
    performance_dict['FN'] = sum(((y_t != 'O') and (y_p == 'O'))
                                 for y_t, y_p in zip(y_true_all, y_pred_all))
    performance_dict['TN'] = sum((y_t == y_p == 'O')
                                 for y_t, y_p in zip(y_true_all, y_pred_all))

    return performance_dict


def sequence_labeling_report(y_true: List[List[str]],
                             y_pred: List[List[str]],
                             digits: int = 2,
                             suffix: bool = False,
                             verbose: int = 1) -> Dict[str, Any]:
    """Build a text report showing the main classification metrics.

    Args:
        y_true: 2d array. Ground truth (correct) target values.
        y_pred: 2d array. Estimated targets as returned by a classifier.
        digits: int. Number of digits for formatting output floating point values.
        suffix:
        verbose:
    Returns:
        report: string. Text summary of the precision, recall, F1 score for each class.

    Examples:
        >>> from kashgari.metrics.sequence_labeling import sequence_labeling_report
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> report = sequence_labeling_report(y_true, y_pred)
                     precision    recall  f1-score   support
        <BLANKLINE>
               MISC       0.00      0.00      0.00         1
                PER       1.00      1.00      1.00         1
        <BLANKLINE>
          micro avg       0.50      0.50      0.50         2
          macro avg       0.50      0.50      0.50         2
        <BLANKLINE>
    """
    true_entities = set(bulk_get_entities(y_true, suffix=suffix))
    pred_entities = set(bulk_get_entities(y_pred, suffix=suffix))

    name_width = 0
    d1 = defaultdict(set)
    d2 = defaultdict(set)
    for e in true_entities:
        d1[e[0]].add((e[1], e[2]))
        name_width = max(name_width, len(e[0]))
    for e in pred_entities:
        d2[e[0]].add((e[1], e[2]))

    last_line_heading = 'macro avg'
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'\n{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'

    report_dic: Dict[str, Any] = {
        'detail': {}
    }

    ps, rs, f1s, s = [], [], [], []
    for type_name, t_true_entities in d1.items():
        t_pred_entities = d2[type_name]
        nb_correct = len(t_true_entities & t_pred_entities)
        nb_pred = len(t_pred_entities)
        nb_true = len(t_true_entities)

        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_true if nb_true > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        report += row_fmt.format(*[type_name, p, r, f1, nb_true], width=width, digits=digits)

        report_dic['detail'][type_name] = {
            "precision": p,
            "recall": r,
            "f1-score": f1,
            "support": nb_true
        }
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(nb_true)

    report += u'\n'

    report_dic['precision'] = np.average(ps, weights=s)
    report_dic['recall'] = np.average(rs, weights=s)
    report_dic['f1-score'] = np.average(f1s, weights=s)
    report_dic['support'] = np.sum(s)

    # compute averages
    report += row_fmt.format('micro avg',
                             precision_score(y_true, y_pred, suffix=suffix),
                             recall_score(y_true, y_pred, suffix=suffix),
                             f1_score(y_true, y_pred, suffix=suffix),
                             np.sum(s),
                             width=width, digits=digits)
    report += row_fmt.format(last_line_heading,
                             np.average(ps, weights=s),
                             np.average(rs, weights=s),
                             np.average(f1s, weights=s),
                             np.sum(s),
                             width=width, digits=digits)
    if verbose:
        print(report)

    return report_dic


if __name__ == "__main__":
    pass
