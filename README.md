# ShanNLP: Shan Natural Language Processing
**experimental project inspired by [PythaiNLP](https://github.com/PyThaiNLP/pythainlp)**

## Current State
- [ ] corpus dict word: 19904 words (60% corvered and need more to collected)

## Word Tokenization method
- [x] maximal_matching
- [x] pythainlp (newmm)

## TODO
- [ ] mining more shan words, poem
- [ ] experiment more method to tokenize
  - [ ] word tokenize
  - [ ] sentent tokenize
  - [ ] subword_tokenize
  - [ ] tokenize with deep learning
- [ ] spelling check
- [ ] pos tagging
- [ ] translation
- [ ] word_vector

## USAGE
```python
# this project using pythainlp dependecy
# - Trie data structure
# - newmm (experimental)

pip install -r requirements.txt
# or pip install pythainlp
```

```python
from shannlp import word_tokenize

# start measure execute time
# start = time.time()

# # Example usage
input_text = "တိူၵ်ႈသွၼ်လိၵ်ႈသင်ၶၸဝ်ႈ တီႈဝဵင်းမိူင်းၶၢၵ်ႇ တႄႇပိုတ်ႇသွၼ်ႁဵၼ်းလိၵ်ႈ ပဵၼ်ပွၵ်ႈၵမ်းႁႅၵ်း မီးသင်ၶၸဝ်ႈ မႃးႁဵၼ်း 56 တူၼ်။"

# default tokenizer engine="mm" (maximal_matching)
print(word_tokenize(input_text))

# end measure execute time
# end = time.time()
# print(end - start)

# output
# ['တိူၵ်ႈ', 'သွၼ်လိၵ်ႈ', 'သင်ၶ', 'ၸဝ်ႈ', ' ', 'တီႈ', 'ဝဵင်း', 'မိူင်းၶၢၵ်ႇ', ' ', 'တႄႇ', 'ပိုတ်ႇ', 'သွၼ်', 'ႁဵၼ်းလိၵ်ႈ', ' ', 'ပဵၼ်', 'ပွၵ်ႈ', 'ၵမ်း', 'ႁႅၵ်း', ' ', 'မီး', 'သင်ၶ', 'ၸဝ်ႈ', ' ', 'မႃး', 'ႁဵၼ်း', ' ', '56', ' ', 'တူၼ်', '။']
# 0.7220799922943115
```
