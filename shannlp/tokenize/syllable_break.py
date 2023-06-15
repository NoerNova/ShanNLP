import re
from typing import Union

TONES = '\u1087|\u1088|\u1038|\u1089|\u108A'
NGA_PYET = '\u103A'
TITS = '\u1030|\u102F'
AYE_SAI = '\u1031|\u1084'
AR = '\u1083'
BREAKS = '\u104A|\u104B'
KAIKURN = '\u1086'
TARNGAKKYAN = '\u102E'
CONSONENTS = '\u1075|\u1076|\u1004|\u1078|\u101E|\u107A|\u1010|\u1011|\u107C|\u1015|\u107D|\u107E|\u1019' \
             '|\u101A|\u101B|\u101C|\u101D|\u1081|\u1022'

toneReg = re.compile(rf"[{TONES}|{BREAKS}]", re.IGNORECASE)
frontReg = re.compile(rf"[{NGA_PYET}|{KAIKURN}|{AR}|{TARNGAKKYAN}]+[^{TONES}]", re.IGNORECASE)
backReg = re.compile(rf"[{AYE_SAI}]+[^{TONES}|{AR}]", re.IGNORECASE)
titsReg = re.compile(rf"[{CONSONENTS}]+[{TITS}]+[{CONSONENTS}]+[^{NGA_PYET}]", re.IGNORECASE)
singleReg = re.compile(rf"[{CONSONENTS}]+[{CONSONENTS}]+[^{NGA_PYET}]", re.IGNORECASE)


def add_space_behind_tones(text: str) -> str:
    if toneReg.search(text):
        text = toneReg.sub(lambda m: m.group() + " ", text)
    return text


def split_match(pattern: re.Pattern, text: str) -> Union[re.Pattern, str]:
    if not pattern.search(text):
        return text
    return pattern.sub(lambda m: m.group()[0] + " " + m.group()[1], text)


def tit_song_join(pattern: re.Pattern, text: str) -> Union[re.Pattern, str]:
    if not pattern.search(text):
        return text
    return pattern.sub(lambda m: m.group()[:2] + " " + m.group()[2:], text)


def consonant_join(pattern: re.Pattern, text: str) -> Union[re.Pattern, str]:
    if not pattern.search(text):
        return text
    return pattern.sub(lambda m: " ".join(list(m.group()[:-1])) + m.group()[-1], text)


def syllable_tokenize(text: str) -> Union[list, None]:
    if not text or not isinstance(text, str):
        return

    text = add_space_behind_tones(text)
    text = split_match(frontReg, text)
    text = split_match(backReg, text)
    text = tit_song_join(titsReg, text)
    text = consonant_join(singleReg, text)

    token_array = text.split(" ")
    return list(filter(None, token_array))
