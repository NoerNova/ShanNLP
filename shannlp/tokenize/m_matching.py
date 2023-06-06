from math import inf  # infinity
from shannlp.corpus import shan_all_corpus
from pythainlp.util.trie import Trie
from typing import List, Optional
import re

_NON_SHAN = re.compile(
    r"""(?x)
    [-a-zA-Z]+|
    \d+([,.]\d+)*|
    [ \t]+|
    \r?\n
    """
)

DEFAULT_WORD_DICT_TRIE = Trie(shan_all_corpus())


def maximal_matching(text: str) -> List[List[Optional[str]]]:
    if not text or not isinstance(text, str):
        return []

    n = len(text)
    text_parts: List[List[Optional[float]]] = [[None] * len(text) for _ in range(len(text))]
    dictionary = DEFAULT_WORD_DICT_TRIE

    for i in range(n):
        for j in range(i, n):
            min_val = 1
            substring = text[i: j + 1]
            if substring in dictionary or substring.isspace() or substring.isdigit():
                if i > 0:
                    prev_col = [
                        text_parts[k][j - 1]
                        for k in range(i)
                        if text_parts[k][j - 1] not in [None, float(inf)]
                    ]
                    if prev_col:
                        min_val = 1 + min(prev_col)

                    text_parts[i][j] = min_val
                else:
                    text_parts[i][j] = 1
            else:
                text_parts[i][j] = inf

    return text_parts


def backtrack(d):
    # eow = len(d) - 1
    word_pos = []

    pre_min = inf
    # focus_row = d[eow]

    num_columns = len(d[0])
    memo_first_index = inf

    for column_index in range(num_columns - 1, -1, -1):
        column_values = [
            row[column_index] for row in d if row[column_index] is not None
        ]
        min_value = min(column_values)
        min_index = [
            i for i, value in enumerate(d) if value[column_index] == min_value
        ][0]

        if min_index < pre_min:
            pre_min = min_index
        else:
            # min_index = pre_min
            continue

        focus_row = d[min_index]

        first_index = None
        last_index = None

        for j in range(len(focus_row)):
            if focus_row[j] is not None and focus_row[j] != float("inf"):
                if j < memo_first_index:
                    last_index = j
                if first_index is None:
                    first_index = j

        memo_first_index = first_index
        word_pos.append((first_index, last_index))

    word_pos.reverse()
    return word_pos


def segment(text: str) -> List[str]:
    tokens = maximal_matching(text)
    tokenized_text = []

    for pos in backtrack(tokens):
        tokenized_text.append(text[pos[0]: pos[1] + 1])

    return tokenized_text
