# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_macros.py
# time: 3:23 下午

import random
import logging
from kashgari.macros import DATA_PATH
from tensorflow.keras.utils import get_file
from kashgari.corpus import ChineseDailyNerCorpus, SMP2018ECDTCorpus

logging.basicConfig(level='DEBUG')

text_x = [
    ['语', '言', '学', '是', '一', '门', '关', '于', '人', '类', '语', '言', '的', '科', '学', '研', '究', '。'],
    ['语', '言', '学', '包', '含', '了', '几', '种', '分', '支', '领', '域', '。'],
    ['在', '语', '言', '结', '构', '研', '究', '与', '意', '义', '研', '究', '之', '间', '存', '在', '一', '个', '重', '要', '的', '主',
     '题', '划', '分', '。'], ['语', '法', '中', '包', '含', '了', '词', '法', '，', '句', '法', '以', '及', '语', '音', '。'],
    ['语', '音', '学', '是', '语', '言', '学', '的', '一', '个', '相', '关', '分', '支', '，', '它', '涉', '及', '到', '语', '音', '与',
     '非', '语', '音', '声', '音', '的', '实', '际', '属', '性', '，', '以', '及', '它', '们', '是', '如', '何', '发', '出', '与', '被',
     '接', '收', '到', '的', '。'],
    ['与', '学', '习', '语', '言', '不', '同', '，', '语', '言', '学', '是', '研', '究', '所', '有', '人', '类', '语', '文', '发', '展',
     '有', '关', '的', '一', '门', '学', '术', '科', '目', '。'],
    ['在', '语', '言', '结', '构', '（', '语', '法', '）', '研', '究', '与', '意', '义', '（', '语', '义', '与', '语', '用', '）', '研',
     '究', '之', '间', '存', '在', '一', '个', '重', '要', '的', '主', '题', '划', '分'],
    ['语', '言', '学', '（', '英', '语', '：', 'l', 'i', 'n', 'g', 'u', 'i', 's', 't', 'i', 'c', 's', '）', '是', '一', '门',
     '关', '于', '人', '类', '语', '言', '的', '科', '学', '研', '究'],
    ['语', '言', '学', '（', '英', '语', '：', 'l', 'i', 'n', 'g', 'u', 'i', 's', 't', 'i', 'c', 's', '）', '是', '一', '门',
     '关', '于', '人', '类', '语', '言', '的', '科', '学', '研', '究'],
    ['语', '言', '学', '（', '英', '语', '：', 'l', 'i', 'n', 'g', 'u', 'i', 's', 't', 'i', 'c', 's', '）', '是', '一', '门',
     '关', '于', '人', '类', '语', '言', '的', '科', '学', '研', '究'],
    ['语', '言', '学', '包', '含', '了', '几', '种', '分', '支', '领', '域', '。'],
    ['在', '语', '言', '结', '构', '（', '语', '法', '）', '研', '究', '与', '意', '义', '（', '语', '义', '与', '语', '用', '）', '研',
     '究', '之', '间', '存', '在', '一', '个', '重', '要', '的', '主', '题', '划', '分']
]

ner_y = [
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
     'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'B-1', 'I-1', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-2', 'O', 'O', 'O', 'B-1', 'I-1', 'I-1', 'O', 'O',
     'O',
     'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
     'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'B-3', 'I-3', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
     'O',
     'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'B-1', 'I-1', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
     'O',
     'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
     'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'B-1', 'I-1', 'I-1', 'O', 'O', 'O', 'O', 'B-1', 'I-1', 'I-1', 'O', 'O', 'O', 'O', 'O',
     'O', 'O',
     'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
     'O', 'O', 'O', 'O', 'O', 'B-2', 'I-2', 'I-2', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'B-3', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'B-3', 'I-3', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
     'O',
     'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]


class TestMacros:
    bert_path = get_file('bert_sample_model',
                         "http://s3.bmio.net/kashgari/bert_sample_model.tar.bz2",
                         cache_dir=DATA_PATH,
                         untar=True)

    w2v_path = get_file('sample_w2v.txt',
                        "http://s3.bmio.net/kashgari/sample_w2v.txt",
                        cache_dir=DATA_PATH)

    chinese_daily = ChineseDailyNerCorpus.load_data('valid')

    smp_corpus = SMP2018ECDTCorpus.load_data('valid')

    # Test data for issue https://github.com/BrikerMan/Kashgari/issues/187
    custom_1 = (text_x, ner_y)

    @classmethod
    def load_labeling_corpus(cls, name=None):
        data_dict = {
            'chinese_daily': cls.chinese_daily,
            'custom_1': cls.custom_1,
        }

        if name is None:
            name = random.choice(list(data_dict.keys()))
        return data_dict[name]

    @classmethod
    def load_classification_corpus(cls, name=None):
        data_dict = {
            'smp_corpus': cls.smp_corpus
        }

        if name is None:
            name = random.choice(list(data_dict.keys()))
        return data_dict[name]


if __name__ == "__main__":
    pass
