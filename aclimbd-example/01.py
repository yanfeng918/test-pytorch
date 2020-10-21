
from transformers import BertTokenizer, BertForSequenceClassification,BertConfig
from transformers import BertModel, BertTokenizer
import torch
import logging
import torch.nn.functional as F
import torch.nn as nn
import time
import tqdm

log = logging.getLogger(__name__)
log.info('start data preparing...')

output_dir = "../bert-base-uncased/"
model = BertForSequenceClassification.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained(output_dir)  # Add specific options if needed

from pathlib import Path

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir is "neg" else 1)

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

batch_size =8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)






if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from transformers import DistilBertForSequenceClassification, AdamW

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optim = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            print(loss)
            loss.backward()
            optim.step()

    model.eval()


