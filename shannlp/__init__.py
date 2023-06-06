from shannlp.tokenize import (
    Tokenizer,
    word_tokenize
)

shan_consonants = "ၵၷၶꧠငၸၹသၺတၻထၼꧣပၽၾပၿႀမယရ႟လꩮဝႁဢ"
shan_vowels = "\u1083\u1062\u1084\u1085\u1031\u1035\u102d\u102e\u102f\u1030\u1086\u1082\u103a\u103d\u103b\u103c"
shan_tone = "\u1087\u1088\u1038\u1089\u108a"
shan_punctuations = "\u104a\u104b\ua9e6"

shan_letters = "".join([shan_consonants, shan_vowels, shan_tone, shan_punctuations])
shan_digits = "႐႑႒႓႔႕႖႗႘႙"

shan_characters = "".join([shan_letters, shan_digits])