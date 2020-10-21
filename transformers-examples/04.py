import logging
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification,BertForSequenceClassification
from transformers import BertTokenizer, BertConfig,BertModel
import torch

from transformers import BertTokenizer, BertForSequenceClassification,BertConfig
import torch
import logging
import csv



output_dir = "./save3/"
#Bert模型示例
model = BertForSequenceClassification.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained(output_dir)  # Add specific options if needed

log = logging.getLogger(__name__)
log.info('start data preparing...')

def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, 'r') as file:
        reader = csv.reader(file, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


tsv = _read_tsv('../data/MRPC/train.tsv')
del(tsv[0])
train_texts_a = [item[3] for item in tsv]
train_texts_b = [item[4] for item in tsv]

train_labels = [int(item[0]) for item in tsv]
print(train_texts_a[0])

tsv_test = _read_tsv('../data/MRPC/train.tsv')
del(tsv_test[0])
test_texts_a = [item[3] for item in tsv_test]
test_texts_b = [item[4] for item in tsv_test]

test_labels = [int(item[0]) for item in tsv_test]




train_encodings = tokenizer(train_texts_a, train_texts_b, truncation=True, padding=True)
test_encodings = tokenizer(test_texts_a, test_texts_b, truncation=True, padding=True)

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, train_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)
# print(model)
ngpu= 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
model.to(device)

train_encodings = tokenizer(train_texts_a, train_texts_b, truncation=True, padding=True)
test_encodings = tokenizer(test_texts_a, test_texts_b, truncation=True, padding=True)

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = IMDbDataset(train_encodings, train_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)
next(iter(test_dataset))

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,       # evaluation dataset
    eval_dataset=test_dataset        # evaluation dataset
)

trainer.train()


from transformers import WEIGHTS_NAME, CONFIG_NAME




import os
# 步骤1：保存一个经过微调的模型、配置和词汇表

#如果我们有一个分布式模型，只保存封装的模型
#它包装在PyTorch DistributedDataParallel或DataParallel中
# model_to_save = model.module if hasattr(model, 'module') else model
# #如果使用预定义的名称保存，则可以使用`from_pretrained`加载
# output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
# output_config_file = os.path.join(output_dir, CONFIG_NAME)
#
# torch.save(model_to_save.state_dict(), output_model_file)
#
# model_to_save.config.to_json_file(output_config_file)
# tokenizer.save_vocabulary(output_dir)
# save_directory = "./save3/"
# tokenizer.save_pretrained(save_directory)
# model.save_pretrained(save_directory)
# 步骤2: 重新加载保存的模型


#GPT模型示例
