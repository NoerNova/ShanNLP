from shannlp.tokenize import (
    Tokenizer,
    word_tokenize,
    syllable_tokenize
)
from shannlp.spell import (
    spell_correct,
    SpellCorrector,
    is_correct_spelling
)

shan_consonants = "ၵၷၶꧠငၸၹသၺတၻထၼꧣပၽၾပၿႀမယရလꩮဝႁဢ"
shan_vowels = "\u1083\u1062\u1084\u1085\u1031\u1035\u102d\u102e\u102f\u1030\u1086\u1082\u103a\u103d\u103b\u103c"
shan_lead_vowels = "\u1084\u1031\u103c"  # ႄ, ေ, ြ
shan_follow_vowels = "\u1083\u1062\u103b"  # ႃ, ၢ, ျ
shan_above_vowels = "\u1085\u1035\u102d\u102e\u1086\u103a"  # ႅ, ဵ, ိ, ီ, ႆ, ်
shan_below_vowels = "\u102f\u1030\u1082\u103d"  # ု, ူ, ႂ, ွ


shan_tonemarks = "\u1087\u1088\u1038\u1089\u108a"
shan_punctuations = "\u104a\u104b\ua9e6"  # ၊, ။, ꧦ

shan_digits = "႐႑႒႓႔႕႖႗႘႙"

shan_letters = "".join([shan_consonants, shan_vowels, shan_tonemarks, shan_punctuations])
shan_characters = "".join([shan_letters, shan_digits])
