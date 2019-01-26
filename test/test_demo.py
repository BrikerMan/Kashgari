# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: test_demo
@time: 2019-01-26

"""
import logging


class TestDemoBase(object):

    def prepare_demo(self):
        self.name = 'base'
        self.code = 500

    def test_hello(self):
        self.prepare_demo()
        print(logging.info('hello, {}'.format(self.name)))
        logging.info('hello, {}'.format(self.name))
        assert self.code == 500


class TestDemo1(TestDemoBase):

    def prepare_demo(self):
        self.name = 'demo1'
        self.code = 500


class TestDemo2(TestDemoBase):

    def prepare_demo(self):
        self.name = 'demo2'
        self.code = 300


if __name__ == '__main__':
    print("hello, world")