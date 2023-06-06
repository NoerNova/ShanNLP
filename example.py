import time
from shannlp import word_tokenize

# to measure time
start = time.time()

# # Example usage
input_text = "ပူၼ်ႉမႃး မိူဝ်ႈဝၼ်းတီႈ 4/6/2023 ယူႇတီႈ ၸဝ်ႈၶူးၸၼ်ႇတႃႇဝႃႇရ (တူႉပီႈၸၼ်) ဢွၼ်ႁူဝ်ၼမ်းၼႃးသေ တႄႇပိုတ်ႇတိူၵ်ႈ သွၼ်လိၵ်ႈသင်ၶၸဝ်ႈ ၾၢႆႇမိူင်း(လိၵ်ႈတႆးၶိုၼ်) တီႈဝတ်ႉဝၢၼ်ႈသဵဝ်ႈ ဢိူင်ႇဝၢၼ်ႈၶုမ်ႉ ၸႄႈဝဵင်းမိူင်းၶၢၵ်ႇ ၸႄႈတွၼ်ႈၵဵင်းတုင် ၸိုင်ႈတႆးပွတ်းဢွၵ်ႇၶူင်း ပွၵ်ႈၵမ်းႁႅၵ်ႈ။"

print(word_tokenize(input_text, keep_whitespace=False))

end = time.time()
print(end - start)
