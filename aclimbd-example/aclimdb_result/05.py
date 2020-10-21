

from transformers import BertModel, BertTokenizer
import torch
import logging
import torch.nn.functional as F
import torch.nn as nn
import time

from tqdm import tqdm

log = logging.getLogger(__name__)
log.info('start data preparing...')

output_dir = "./aclimdb_result/"
# Bert模型示例
model = BertModel.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained(output_dir)  # Add specific options if needed

from pathlib import Path

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir == "neg" else 1)

    return texts, labels

train_texts, train_labels = read_imdb_split('../data/aclImdb/train')
test_texts, test_labels = read_imdb_split('../data/aclImdb/test')


from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)


train_encodings = tokenizer(train_texts, truncation=True, padding=True,max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True,max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True,max_length=512)


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
val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)

from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(device)
model.train()

batch_size =4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

class Bert_Fc_Model(nn.Module):
    def __init__(self):
        super(Bert_Fc_Model, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.fc = nn.Linear(768, batch_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids,attention_mask,token_type_ids):
        out = model(input_ids, attention_mask, token_type_ids)
        out = self.dropout(out[0])
        pooled = F.avg_pool2d(out, (out.shape[1], 1)).squeeze(1)  # [batch size, embedding_dim]
        out = self.fc(pooled)
        return out

bert_model = Bert_Fc_Model()


def train():

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(bert_model.parameters(), lr=5e-5)
    correct_number = 0
    time_start = time.time()  # 开始时间

    correct_total_number =0
    for epoch in range(3):
        for i, batch in enumerate(tqdm(train_loader)):
            # batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            label = batch['labels'].to(device)
            out = bert_model(input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids)
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

    save_directory = "./aclimdb_result/"
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)
    torch.save(bert_model.state_dict(), "./aclimdb_result/embedding-{}.th".format(756))
    # model.load_state_dict(torch.load("embedding-{}.th".format(EMBEDDING_SIZE)))

def eval():
    correct_total_number =0

    bert_model.load_state_dict(torch.load("./aclimdb_result/embedding-{}.th".format(756)))

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

