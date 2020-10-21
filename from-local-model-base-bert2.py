from transformers import BertModel, BertTokenizer


output_dir = "./bert-base-uncased/"
model = BertModel.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained(output_dir)  # Add specific options if needed
print(model.config)


paraphrase = tokenizer("hello wrold", return_tensors="pt")

print(paraphrase)
