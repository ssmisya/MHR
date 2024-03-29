from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from m3apo.vcd.experiments.eval.language_dict import language_dict,nllb_200_distilled_600M_language_dict
from m3apo.utils.utils import load_json_file, write_json_file


## settings

source_lang="en"
target_lang="zh"
source_lang = nllb_200_distilled_600M_language_dict[language_dict[source_lang]['full_name']]
target_lang = nllb_200_distilled_600M_language_dict[language_dict[target_lang]['full_name']]
# translate_model = AutoModelForSeq2SeqLM.from_pretrained("/mnt/petrelfs/songmingyang/songmingyang/model/others/nllb-200-distilled-600M")
# translate_model = translate_model.eval()
# translate_tokenizer = AutoTokenizer.from_pretrained("/mnt/petrelfs/songmingyang/songmingyang/model/others/nllb-200-distilled-600M")


# print(dir(translate_model))

from transformers import AutoTokenizer, M2M100ForConditionalGeneration

# model = M2M100ForConditionalGeneration.from_pretrained("/mnt/petrelfs/songmingyang/songmingyang/model/others/nllb-200-distilled-600M")
# tokenizer = AutoTokenizer.from_pretrained("/mnt/petrelfs/songmingyang/songmingyang/model/others/nllb-200-distilled-600M")
# tokenizer.source_lang = source_lang
# # tokenizer.target_lang = target_lang
# # text_to_translate = ["Life is like a box of chocolates 1","I am groot 2","I want to boost this translation model 3","what can I do? 4"]
# text_to_translate = "Life is like a box of chocolates 1"
# input = tokenizer(text_to_translate, return_tensors='pt', padding='longest', truncation=True,max_length=512).to(model.device)
# output = model.generate(**input, forced_bos_token_id=tokenizer.lang_code_to_id[target_lang])
# output_text = tokenizer.batch_decode(output,skip_special_tokens=True)
# print(output_text)
# print(input)

source_file = "/mnt/petrelfs/songmingyang/songmingyang/data/mm/annotation/hadpo-data/hadpo/llava-v1.5/desc_data.json"
target_file = "/mnt/petrelfs/songmingyang/songmingyang/data/mm/annotation/hadpo-data/hadpo/llava-v1.5/multilingual/desc_fr.json"

data_a = load_json_file(source_file)
data_b = load_json_file(source_file)

assert len(data_a) == len(data_b)
for k,v in data_a.items():
    item_b = data_b[k]
    assert len(item_b["chosen"]) == len(v["chosen"])
    assert len(item_b["rejected"]) == len(v["rejected"])