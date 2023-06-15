__all__ = [
    "Tokenizer",
    "word_tokenize",
    "DEFAULT_WORD_DICT_TRIE",
    "DEFAULT_WORD_TOKENIZE_ENGINE",
    "syllable_tokenize"
]

from shannlp.corpus import shan_all_corpus
from pythainlp.util.trie import Trie

DEFAULT_WORD_TOKENIZE_ENGINE = "mm"

DEFAULT_WORD_DICT_TRIE = Trie(shan_all_corpus())

from shannlp.tokenize.core import Tokenizer, word_tokenize
from shannlp.tokenize.syllable_break import syllable_tokenize
