# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: base_tokenizer.py
# time: 11:24 上午

from typing import List


class Tokenizer:
    """
    Abstract base class for all implemented tokenizer.
    """

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into token sequence
        Args:
            text: target text sample

        Returns:
            List of tokens in this sample
        """
        return text.split(' ')
