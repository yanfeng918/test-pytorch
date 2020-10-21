import transformers
MODEL_PATH = "./bert-base-uncased/"
# a.通过词典导入分词器
tokenizer = transformers.BertTokenizer.from_pretrained(r"./bert-base-uncased/vocab.txt")
# b. 导入配置文件
model_config = transformers.BertConfig.from_pretrained("./bert-base-uncased/config.json")
# 修改配置
model_config.output_hidden_states = True
model_config.output_attentions = True
# 通过配置和路径导入模型
model = transformers.BertModel.from_pretrained(MODEL_PATH,config = model_config)
# print(model)
print(model.config)

paraphrase = tokenizer("hello wrold", return_tensors="pt")

print(paraphrase)
