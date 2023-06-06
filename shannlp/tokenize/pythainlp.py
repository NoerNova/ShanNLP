from typing import List
from pythainlp.tokenize import Tokenizer
from shannlp.corpus import shan_words

_word = Tokenizer(shan_words())


def word_tokenize(sent: str) -> List[str]:
    return _word.word_tokenize(sent)
