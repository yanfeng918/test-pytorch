import torch
flag = torch.cuda.is_available()
print(flag)

ngpu= 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
print(torch.__version__)
print(torch.cuda.get_device_name(0))
print(torch.rand(3,3).cuda())


from transformers import BertTokenizer, BertForSequenceClassification,BertConfig
import torch
import logging
import csv


MODEL_PATH = "/home/yanfeng/PycharmProjects/test/bert-base-uncased"
# a.通过词典导入分词器
tokenizer = BertTokenizer.from_pretrained(r"/home/yanfeng/PycharmProjects/test/bert-base-uncased/vocab.txt")
# b. 导入配置文件
model_config = BertConfig.from_pretrained("/home/yanfeng/PycharmProjects/test/bert-base-uncased/config.json")
# 修改配置
model_config.output_hidden_states = True
model_config.output_attentions = True
# 通过配置和路径导入模型
model = BertForSequenceClassification.from_pretrained(MODEL_PATH,config = model_config)

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


tsv = _read_tsv('./data/MRPC/train.tsv')
del(tsv[0])
train_texts_a = [item[3] for item in tsv]
train_texts_b = [item[4] for item in tsv]

train_labels = [int(item[0]) for item in tsv]
print(train_texts_a[0])

tsv_test = _read_tsv('./data/MRPC/train.tsv')
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
