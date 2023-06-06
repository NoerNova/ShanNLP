# -*- coding: utf-8 -*-
# Copyright (C) 2016-2023 PyThaiNLP Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Trie data structure.

Designed to use for tokenizer's dictionary, but can be for other purposes.
"""
from typing import Iterable, List, Union


class Trie:
    class Node(object):
        __slots__ = "end", "children"

        def __init__(self):
            self.end = False
            self.children = {}

    def __init__(self, words: Iterable[str]):
        self.words = set(words)
        self.root = Trie.Node()

        for word in words:
            self.add(word)

    def add(self, word: str) -> None:
        """
        Add a word to the trie.
        Spaces in front of and following the word will be removed.

        :param word:
        """
        word = word.strip()
        self.words.add(word)
        cur = self.root
        for ch in word:
            child = cur.children.get(ch)
            if not child:
                child = Trie.Node()
                cur.children[ch] = child
            cur = child
        cur.end = True

    def remove(self, word: str) -> None:
        """
        Remove a word from the trie.
        If the word is not found, do nothing.

        :param word:
        """
        # remove from set first
        if word not in self.words:
            return
        self.words.remove(word)
        # then remove from nodes
        parent = self.root
        data = []  # track path to leaf
        for ch in word:
            child = parent.children[ch]
            data.append((parent, child, ch))
            parent = child
        # remove the last one
        child.end = False
        # prune up the tree
        for parent, child, ch in reversed(data):
            if child.end or child.children:
                break
            del parent.children[ch]  # remove from parent dict

    def prefixes(self, text: str) -> List[str]:
        """
        List all possible words from first sequence of characters in a word.

        :param str text: a word
        :return: a list of possible words
        :rtype: List[str]
        """
        res = []
        cur = self.root
        for i, ch in enumerate(text):
            node = cur.children.get(ch)
            if not node:
                break
            if node.end:
                res.append(text[: i + 1])
            cur = node
        return res

    def __contains__(self, key: str) -> bool:
        return key in self.words

    def __iter__(self) -> Iterable[str]:
        yield from self.words

    def __len__(self) -> int:
        return len(self.words)


def dict_trie(dict_source: Union[str, Iterable[str], Trie]) -> Trie:
    """
    Create a dictionary trie from a file or an iterable.

    :param str|Iterable[str]|pythainlp.util.Trie dict_source: a path to
        dictionary file or a list of words or a pythainlp.util.Trie object
    :return: a trie object
    :rtype: pythainlp.util.Trie
    """
    trie = None

    if isinstance(dict_source, str) and len(dict_source) > 0:
        # dict_source is a path to dictionary text file
        with open(dict_source, "r", encoding="utf8") as f:
            _vocabs = f.read().splitlines()
            trie = Trie(_vocabs)
    elif isinstance(dict_source, Iterable) and not isinstance(dict_source, str):
        # Note: Since Trie and str are both Iterable,
        # so the Iterable check should be here, at the very end,
        # because it has less specificality
        trie = Trie(dict_source)
    else:
        raise TypeError(
            "Type of dict_source must be pythainlp.util.Trie, "
            "or Iterable[str], or non-empty str (path to source file)"
        )

    return trie
