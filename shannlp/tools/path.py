import os

from shannlp import __file__ as shannlp_file


def get_shannlp_path() -> str:
    return os.path.dirname(shannlp_file)
