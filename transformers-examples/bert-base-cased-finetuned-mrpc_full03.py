from transformers import BertModel, BertTokenizer

import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification,BertConfig
import torch
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import csv
from torch.utils.data import DataLoader,dataset
import time
import torch.nn.functional as F
from tqdm import tqdm

# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")


MODEL_PATH = "../bert-base-uncased/"
# a.通过词典导入分词器
tokenizer = BertTokenizer.from_pretrained(r"../bert-base-uncased/vocab.txt")
# b. 导入配置文件
model_config = BertConfig.from_pretrained("../bert-base-uncased/config.json")
# 修改配置
model_config.output_hidden_states = True
model_config.output_attentions = True
# 通过配置和路径导入模型
model = BertModel.from_pretrained(MODEL_PATH,config = model_config)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

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



optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)


batch_size =32
# test_dataset = IMDbDataset(test_encodings, test_labels)

class Bert_Fc_Model(nn.Module):
    def __init__(self):
        print("init________")
        super(Bert_Fc_Model, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.fc = nn.Linear(768, batch_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids,attention_mask,token_type_ids):
        out = model(input_ids, attention_mask, token_type_ids)
        # out = input_str
        # print("out.size() = ", out.size())
        # out_0 = out[0]
        # print("out[0].size() = ",out_0.size())
        # out_0 = self.dropout(out_0)
        # print("out[0].size() = ",out_0.size())
        #
        # out = out[1]
        # print("out[1].size() = ", out.size())
        # batch*seqs*hidden
        out = self.dropout(out[0])
        pooled = F.avg_pool2d(out, (out.shape[1], 1)).squeeze(1)  # [batch size, embedding_dim]
        out = self.fc(pooled)
        return out

bert_model = Bert_Fc_Model()
bert_model.train().to(device)

# print(bert_model)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def train():

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(bert_model.parameters(), lr=5e-5)
    correct_number = 0
    time_start = time.time()  # 开始时间

    correct_total_number =0
    for epoch in range(10):
        for i, batch in enumerate(tqdm(train_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            label = batch["labels"]

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            label = batch['labels'].to(device)
            out = bert_model(input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids)
            # batch.pop("labels")
            # outputs = model(**batch)
            # loss = outputs[0]
            # loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()
            # if i % 10 == 0:
            #     print(f"loss: {loss}")
            #print("out = ",out)
            #         print("out = ",out)
            # 清零梯度
            optim.zero_grad()
            loss = F.cross_entropy(out, label)
            # 反向传播
            loss.backward()
            # 优化参数
            optim.step()
            # 计算准确率
            predict = torch.argmax(out,dim=-1)
            # print("predict label = ",predict,"label = ",label)
            # if label == predict:
            correct_number += torch.sum(torch.eq(label,predict)).item()
            correct_total_number += torch.sum(torch.eq(label,predict)).item()
            correct_number = 0
            # print("acc = ",correct_number/ batch_size)
            # print("loss = ", loss)
        # print("acc = ", str(round(correct_number / batch_size * 100, 2)), r"%")
        # # 清空计分
        # correct_number = 0
        print("acc = ",correct_total_number/ len(train_dataset))
        print("总耗时：", time.time() - time_start)
        correct_total_number=0

    save_directory = "./mrpc/"
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)

    torch.save(bert_model.state_dict(), "./mrpc/embedding-{}.th".format(756))
    torch.save(bert_model.state_dict(), "./embedding-{}.th".format(756))
    # model.load_state_dict(torch.load("embedding-{}.th".format(EMBEDDING_SIZE)))


def eval():
    correct_total_number =0
    output_dir = "./mrpc/"
    # Bert模型示例
    model = BertModel.from_pretrained(output_dir)
    tokenizer = BertTokenizer.from_pretrained(output_dir)  # Add specific options if needed
    bert_model.load_state_dict(torch.load("embedding-{}.th".format(756)))

    bert_model.eval().to(device)

    for i, batch in enumerate(tqdm(train_loader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        label = batch['labels'].to(device)
        out = bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        predict = torch.argmax(out, dim=-1)
        correct_total_number += torch.sum(torch.eq(label, predict)).item()

    print("acc = ", correct_total_number / len(train_dataset))

if __name__ == "__main__":
    eval()