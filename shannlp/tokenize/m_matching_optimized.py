import re
from typing import List, Optional
from math import inf
from pythainlp.util.trie import Trie
from shannlp.corpus import shan_all_corpus

# Constants
# NON_SHAN = re.compile(r"(?x)[-a-zA-Z]+|\d+([,.]\d+)*|[ \t]+|\r?\n")
SHAN_UNICODE_RANGE = re.compile(r"[\u1000-\u109F]")
DEFAULT_WORD_DICT_TRIE = Trie(shan_all_corpus())


def maximal_matching(text: str, custom_dict: Trie) -> List[List[Optional[float]]]:
    n = len(text)
    text_parts = [[None] * n for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            substring = text[i : j + 1]
            if substring in custom_dict or not SHAN_UNICODE_RANGE.search(substring):
                min_val = 1
                if i > 0:
                    prev_col = [
                        text_parts[k][j - 1]
                        for k in range(i)
                        if text_parts[k][j - 1] not in [None, inf]
                    ]
                    if prev_col:
                        min_val += min(prev_col)
                text_parts[i][j] = min_val
            else:
                text_parts[i][j] = inf

    return text_parts


def backtrack(d: List[List[Optional[float]]]) -> List[tuple[int, int]]:
    word_pos = []
    num_columns = len(d[0])
    current_end = num_columns - 1

    while current_end >= 0:
        col_values = [
            row[current_end]
            for row in d
            if row[current_end] is not None and row[current_end] != inf
        ]
        if not col_values:
            current_end -= 1
            continue

        min_value = min(col_values)
        start = next(i for i, row in enumerate(d) if row[current_end] == min_value)

        word_pos.append((start, current_end))
        current_end = start - 1

    return list(reversed(word_pos))


def segment(text: str, custom_dict: Trie = DEFAULT_WORD_DICT_TRIE) -> List[str]:
    if not text or not isinstance(text, str):
        return []

    if not custom_dict:
        custom_dict = DEFAULT_WORD_DICT_TRIE

    tokens = maximal_matching(text, custom_dict)
    return [text[start : end + 1] for start, end in backtrack(tokens)]
