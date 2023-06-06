import time
from shannlp import Tokenizer, word_tokenize

# to measure time
start = time.time()

# # Example usage

input_text = "တိူၵ်ႈသွၼ်လိၵ်ႈသင်ၶၸဝ်ႈ တီႈဝဵင်းမိူင်းၶၢၵ်ႇ တႄႇပိုတ်ႇသွၼ်ႁဵၼ်းလိၵ်ႈ ပဵၼ်ပွၵ်ႈၵမ်းႁႅၵ်း မီးသင်ၶၸဝ်ႈ မႃးႁဵၼ်း 56 တူၼ်။"

tokenizer = Tokenizer()

print(word_tokenize(input_text))

end = time.time()
print(end - start)
