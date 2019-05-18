# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_pre_processor.py
# time: 2019-05-18 13:13

import logging
import unittest
from kashgari.pre_processor import PreProcessor

logging.basicConfig(level=logging.DEBUG)

LABELING_X_DATA = [
    ['后', '者', '曾', '明', '确', '宣', '布', '苏', '将', '归', '还', '齿', '舞', '和', '色', '丹', '两', '岛', '。'],
    ['公', '告', '强', '烈', '谴', '责', '阿', '族', '非', '法', '武', '装', '制', '造', '的', '恐', '怖', '活',
     '动', '，', '重', '申', '政', '府', '将', '采', '取', '坚', '决', '措', '施', '保', '障', '科', '索', '沃',
     '各', '民', '族', '居', '民', '和', '平', '、', '安', '宁', '的', '生', '活', '。'],
    ['战', '友', '们', '在', '打', '扫', '战', '场', '时', '看', '见', '他', '那', '张', '血', '肉', '模', '糊',
     '的', '脸', '，', '以', '为', '连', '长', '已', '经', '牺', '牲', '，', '就', '迅', '速', '撤', '出', '了',
     '阵', '地', '。']]

LABELING_Y_DATA = [
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'B-LOC',
     'I-LOC', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
     'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC',
     'I-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
     'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
]


CLASSIFICATION_X_DATA = [
    ['看', '动物', '卡通'],
    ['厦门', '到', '福建', '建阳', '的', '火车', '是', '几点', '呢'],
    ['这么', '任性'], ['中央', '四', '频道', '火箭', '兜风'],
    ['帮', '我', '来', '一首', '兰陵王', '。'], ['朗读', '游子吟', '。'],
    ['这', '不是', '你', '的', '名字', '吗', '😂'], ['淋病', '怎么办'],
    ['发邮件', '给', '莹莹', '说', '我', '好困', '我', '想', '睡觉'],
    ['帮', '我', '调转', '到', '风云', '音乐频道']
]

CLASSIFICATION_Y_DATA = [
    'video',
    'train',
    'chat',
    'epg',
    'poetry',
    'poetry',
    'chat',
    'health',
    'email',
    'tvchannel'
]


class TestPreProcessor(unittest.TestCase):

    def test_prepare_labeling_dicts_if_need(self):
        p = PreProcessor()
        p.prepare_labeling_dicts_if_need(LABELING_X_DATA, LABELING_Y_DATA)
        assert len(p.token2idx) == 100
        assert len(p.label2idx) == 3
        assert p.label2idx == {'O': 0, 'B-LOC': 1, 'I-LOC': 2}
        assert p.idx2token[12] == '战'

    def test_save_and_load(self):
        old_p = PreProcessor()
        old_p.prepare_labeling_dicts_if_need(LABELING_X_DATA, LABELING_Y_DATA)
        old_p.save_dicts('./saved_preprocessor')

        p = PreProcessor()
        p.load_cached_dicts('./saved_preprocessor')

        assert len(p.token2idx) == 100
        assert len(p.label2idx) == 3
        assert p.label2idx == {'O': 0, 'B-LOC': 1, 'I-LOC': 2}
        assert p.idx2token[12] == '战'

    def test_numerize_and_reverse_numerize(self):
        p = PreProcessor()
        p.prepare_labeling_dicts_if_need(LABELING_X_DATA, LABELING_Y_DATA)
        num_x_data_0 = p.numerize_token_sequence(LABELING_X_DATA[0])
        num_y_data_0 = p.numerize_label_sequence(LABELING_Y_DATA[0])
        assert num_x_data_0 == [13, 14, 15, 16, 17, 18, 19, 20, 7, 21, 22, 23, 24, 8, 25, 26, 27, 28, 4]
        assert num_y_data_0 == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 2, 0, 1, 2, 0, 0, 0]

        assert LABELING_Y_DATA[0] == p.reverse_numerize_label_sequence(num_y_data_0)
        assert LABELING_Y_DATA[0][:10] == p.reverse_numerize_label_sequence(num_y_data_0, 10)


if __name__ == "__main__":
    print("Hello world")
