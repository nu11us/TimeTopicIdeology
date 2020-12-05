import json
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.nn.functional as F
import datetime


LEARNING_RATE = 3e-6
WEIGHT_DECAY = 1e-2
BATCH_SIZE = 4
NUM_WORKERS = 1
LR_STEP = 1
GAMMA = 0.3

ENCODING = { 
    "anarchism": [0,0],
    "anarcho_capitalism": [1,1],
    "antiwork": [2,0],
    "breadtube": [3,0],
    "chapotraphouse": [4,0],
    "communism": [5,0],
    "completeanarchy": [6,0],
    "conservative": [7,1],
    "cringeanarchy": [8,1],
    "democraticsocialism": [9,0],
    "esist": [10,0],
    "fullcommunism": [11,0],
    "goldandblack": [12,1],
    "jordanpeterson": [13,1],
    "keep_track": [14,0],
    "latestagecapitalism": [15,0],
    "latestagesocialism": [16,1],
    "liberal": [17,0],
    "libertarian": [18,1],
    "neoliberal": [19,0],
    "onguardforthee": [20,1],
    "ourpresident": [21,1],
    "political_revolution": [22,0],
    "politicalhumor": [23,0],
    "politics": [24,0],
    "progressive": [25,0],
    "republican": [26,1],
    "sandersforpresident": [27,0],
    "selfawarewolves": [28,0],
    "socialism": [29,0],
    "the_donald": [30,1],
    "the_mueller": [31,0],
    "thenewright": [32,1],
    "voteblue": [33,0],
    "wayofthebern": [34,0],
    "yangforpresidenthq": [35,0]
}


class PolReddit(torch.utils.data.Dataset):
    def __init__(self, test = False):
        super(PolReddit).__init__() 
        self.path = "subdata/2016.tsv"
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.test = test
        self.df = pd.read_csv(
                self.path,
                sep='\t',
                header = 0,
                usecols=['subreddit', 'created_utc', 'body'])
        self.body = self.df['body']
        self.subr = self.df['subreddit']

    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(
            self.body[index],
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )
        t = self.df['created_utc'][index]
        d = datetime.datetime.fromtimestamp(int(t))
        v = torch.tensor([int(d.year), int(d.month), int(d.day), int(d.hour), int(d.second)], dtype=torch.float)

        if self.test:
            return {
                'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
                'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
                'time': v
            }
        else:
            return {
                'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
                'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
                'targets': torch.tensor(ENCODING[self.subr[index]][1], dtype=torch.long),
                'time': v
            }

    def __len__(self):
        return len(self.df)

def train(trainloader, num_epochs=10, auto_val=False):
    for epoch in range(num_epochs):
        run_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            inputs, labels, mask, time = data['ids'].to(
                device), data['targets'].to(device), data['mask'].to(device), data['time'].to(device)
            outputs = model(inputs, mask, time)
            loss_val = loss(outputs, labels)
            loss_val.backward()
            total += BATCH_SIZE
            optimizer.step()
            run_loss += loss_val.item()
            correct += (outputs.data.max(1).indices == labels).sum().item()
            if i % 100 == 0:
                print('Loss/train', (run_loss/total), i*BATCH_SIZE + epoch*len(trainloader))
                print('Accuracy/train', (correct/total), i*BATCH_SIZE + epoch*len(trainloader))
                run_loss = 0.0
                correct = 0.0
                total = 0.0
        if auto_val:
            test(val_loader, epoch)
        scheduler.step()

def test(testloader, count=1):
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        i = 0
        for data in testloader:
            inputs, labels, mask, time = data['ids'].to(
                device), data['targets'].to(device), data['mask'].to(device), data['time'].to(device)
            outputs = model(inputs, mask, time)
            total += BATCH_SIZE
            correct += (outputs.data.max(1).indices == labels).sum().item()
            i += 1
            if i%100 == 0 and i!=0:
                print(i, correct/total)
        print('Accuracy/test', correct/total, count)

class Timetron(nn.Module):
    def __init__(self):
        super(Timetron, self).__init__() 
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.l0 = nn.Linear(768, 256)
        self.drop = nn.Dropout(p=0.25)
        self.l1 = nn.Linear(261, 100)
        self.l2 = nn.Linear(100, 2)

    def forward(self, x, y, t):
        x = self.bert.forward(input_ids=x, attention_mask=y)
        x = self.drop(F.relu(self.l0(x[1])))
        x = torch.cat((x,F.tanh(t)),1)
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

if __name__ == "__main__":
    model = Timetron()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=GAMMA)
    model.to(device)
    loss.to(device)


    data = PolReddit()
    dump_data, kept_data = torch.utils.data.random_split(data, [int(0.99*len(data)), len(data)-int(0.99*len(data))])
    train_data, val_data = torch.utils.data.random_split(kept_data, [int(0.8*len(kept_data)), len(kept_data)-int(0.8*len(kept_data))])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, pin_memory=True, shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_data, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, pin_memory=True, shuffle=True)
    train(train_loader)