import re
from typing import List

from shannlp.tokenize import DEFAULT_WORD_TOKENIZE_ENGINE

# from pythainlp.tokenize._utils import (apply_postprocessors, strip_whitespace)

# customize from pythainlp
from shannlp.tokenize._utils import (apply_postprocessors, strip_whitespace)


def word_tokenize(
        text: str,
        engine: str = DEFAULT_WORD_TOKENIZE_ENGINE,
        keep_whitespace: bool = True,
) -> List[str]:
    if not text or not isinstance(text, str):
        return []

    if engine == "mm":
        from shannlp.tokenize.m_matching import segment

        segments = segment(text)
    elif engine == "pythainlp":
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
    if not keep_whitespace:
        postprocessors.append(strip_whitespace)

    segments = apply_postprocessors(segments, postprocessors)

    return segments


class Tokenizer:

    def __init__(
            self,
            engine: str = "mm",
            keep_whitespace: bool = True
    ):
        self.__engine = engine

        if self.__engine not in ["mm"]:
            raise NotImplementedError(
                f"""
                    The Tokenizer class is not support {self.__engine}
                """
            )
        self.__keep_whitespace = keep_whitespace

    def word_tokenize(self, text: str) -> List[str]:
        return word_tokenize(
            text,
            engine=self.__engine,
            keep_whitespace=self.__keep_whitespace
        )

    def set_tokenize_engine(self, engine: str) -> None:
        self.__engine = engine
