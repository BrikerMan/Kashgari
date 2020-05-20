# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: multi_label_classification.py
# time: 6:33 下午

from typing import Dict, Any, TYPE_CHECKING

import numpy as np
from sklearn import metrics

from kashgari.types import MultiLabelClassificationLabelVar

if TYPE_CHECKING:
    from kashgari.utils import MultiLabelBinarizer


def multi_label_classification_report(y_true: MultiLabelClassificationLabelVar,
                                      y_pred: MultiLabelClassificationLabelVar,
                                      *,
                                      binarizer: 'MultiLabelBinarizer',
                                      digits: int = 4,
                                      verbose: int = 1) -> Dict[str, Any]:
    y_pred_b = binarizer.transform(y_pred)
    y_true_b = binarizer.transform(y_true)

    report_dic: Dict = {}
    details: Dict = {}

    rows = []
    ps, rs, f1s, s = [], [], [], []
    for c_index, c in enumerate(binarizer.classes):
        precision = metrics.precision_score(y_true_b[:, c_index], y_pred_b[:, c_index])
        recall = metrics.recall_score(y_true_b[:, c_index], y_pred_b[:, c_index])
        f1 = metrics.f1_score(y_true_b[:, c_index], y_pred_b[:, c_index])
        support = len(np.where(y_true_b[:, c_index] == 1)[0])
        details[c] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }

        rows.append((c, precision, recall, f1, support))
        ps.append(precision)
        rs.append(recall)
        f1s.append(f1)
        s.append(support)

    report_dic['precision'] = np.average(ps, weights=s)
    report_dic['recall'] = np.average(rs, weights=s)
    report_dic['f1-score'] = np.average(f1s, weights=s)
    report_dic['support'] = np.sum(s)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers) + '\n'

    report = head_fmt.format('', *headers, width=20)

    row_fmt = '{:>{width}s}  {:>9.{digits}f} {:>9.{digits}f} {:>9.{digits}f} {:>9}\n'

    for row in rows:
        report += row_fmt.format(*row, width=20, digits=digits)

    # compute averages
    report += row_fmt.format('macro avg',
                             np.average(ps, weights=s),
                             np.average(rs, weights=s),
                             np.average(f1s, weights=s),
                             np.sum(s),
                             width=20, digits=digits)

    report_dic['detail'] = details
    print(report)

    return report_dic


if __name__ == "__main__":
    pass
