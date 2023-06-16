# original from pythainlp
# custom use for shannlp

import re
from typing import List, Callable

_DIGITS_WITH_SEPARATOR = re.compile(r"(\d+[.:,])+(\d+)")


def apply_postprocessors(
        segments: List[str], postprocessors: List[Callable[[List[str]], List[str]]]
) -> List[str]:
    for func in postprocessors:
        segments = func(segments)

    return segments


def rejoin_formatted_num(segments: List[str]) -> List[str]:
    original = "".join(segments)
    matching_results = _DIGITS_WITH_SEPARATOR.finditer(original)
    tokens_joined = []
    pos = 0
    segment_idx = 0

    match = next(matching_results, None)
    while segment_idx < len(segments) and match:
        is_span_beginning = pos >= match.start()
        token = segments[segment_idx]
        if is_span_beginning:
            connected_token = ""
            while pos < match.end() and segment_idx < len(segments):
                connected_token += segments[segment_idx]
                pos += len(segments[segment_idx])
                segment_idx += 1

            tokens_joined.append(connected_token)
            match = next(matching_results, None)
        else:
            tokens_joined.append(token)
            segment_idx += 1
            pos += len(token)
    tokens_joined += segments[segment_idx:]

    return tokens_joined


def strip_whitespace(segments: List[str]) -> List[str]:
    segments = [token.strip(" ") for token in segments if token.strip(" ")]

    return segments
