import re
from typing import List, Iterable, Union

from shannlp.tokenize import DEFAULT_WORD_TOKENIZE_ENGINE, DEFAULT_WORD_DICT_TRIE

# from pythainlp.tokenize._utils import (apply_postprocessors, strip_whitespace)

# customize from pythainlp
from shannlp.tokenize._utils import (apply_postprocessors, strip_whitespace, rejoin_formatted_num)

from pythainlp.util.trie import Trie, dict_trie


def word_tokenize(
        text: str,
        custom_dict: Trie = None,
        engine: str = DEFAULT_WORD_TOKENIZE_ENGINE,
        keep_whitespace: bool = True,
        join_broken_num: bool = True
) -> List[str]:
    if not text or not isinstance(text, str):
        return []

    if engine == "mm":
        from shannlp.tokenize.m_matching import segment

        segments = segment(text)
    elif engine == "newmm":
        from shannlp.tokenize.pythainlp import word_tokenize

        segments = word_tokenize(text)
    elif engine == "whitespace":
        segments = re.split(r" +", text, re.U)
    elif engine == "whitespce+newline":
        segments = text.split()
    else:
        raise ValueError(
            f"""Tokenizer \"{engine}" not found."""
        )

    postprocessors = []
    if join_broken_num:
        postprocessors.append(rejoin_formatted_num)

    if not keep_whitespace:
        postprocessors.append(strip_whitespace)

    segments = apply_postprocessors(segments, postprocessors)

    return segments


class Tokenizer:

    def __init__(
            self,
            custom_dict: Union[Trie, Iterable[str], str] = None,
            engine: str = "mm",
            keep_whitespace: bool = True,
            join_broken_num: bool = True
    ):
        self.__tire_dict = None
        if custom_dict:
            self.__tire_dict = dict_trie(custom_dict)
        else:
            self.__tire_dict = DEFAULT_WORD_DICT_TRIE
        self.__engine = engine

        if self.__engine not in ["mm", "newmm"]:
            raise NotImplementedError(
                f"""
                    The Tokenizer class is not support {self.__engine}
                """
            )
        self.__keep_whitespace = keep_whitespace
        self.__join_broken_num = join_broken_num

    def word_tokenize(self, text: str) -> List[str]:
        return word_tokenize(
            text,
            custom_dict=self.__tire_dict,
            engine=self.__engine,
            keep_whitespace=self.__keep_whitespace,
            join_broken_num=self.__join_broken_num
        )

    def set_tokenize_engine(self, engine: str) -> None:
        self.__engine = engine
