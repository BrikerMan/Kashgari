# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: jieba_tokenizer.py
# time: 11:54 上午

from typing import List, Any

from kashgari.tokenizers.base_tokenizer import Tokenizer


class JiebaTokenizer(Tokenizer):
    """
    Jieba tokenizer
    """

    def __init__(self) -> None:
        try:
            import jieba
            self._jieba = jieba
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Jieba module not found, please install use `pip install jieba`")

    def tokenize(self, text: str, **kwargs: Any) -> List[str]:
        """
        Tokenize text into token sequence
        Args:
            text: target text sample

        Returns:
            List of tokens in this sample
        """

        return list(self._jieba.cut(text, **kwargs))
