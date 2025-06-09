import pandas as pd
import numpy as np
import torch
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large")
id2label = {0: "NONSEXIST", 1: "SEXIST"}
label2id = {"NONSEXIST": 0, "SEXIST": 1}
model = AutoModelForSequenceClassification.from_pretrained('model_r4')
optimizer = AdamW(model.parameters(), lr=1e-6)
num_epochs = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_dataset():
    # train_data = pd.read_csv('train_weibo_text.csv')
    # test_data = pd.read_csv('test_weibo_text.csv')
    dataset = load_dataset('csv', data_files={'train': 'weibo_emoji_data/train_weibo_emoji.csv', 'test': 'weibo_emoji_data/test_weibo_emoji.csv'})
    return dataset

def tokenize_function(example):
    return tokenizer(example['comment_text'],padding=True,truncation=True,return_tensors="pt")

def train(model, train_dataloader, optimizer, scheduler, num_epochs, num_training_steps):
    progress_bar = tqdm(range(num_training_steps))
    model.train()

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)


def eval(model, test_dataloader):
    acc = evaluate.load('accuracy')
    f1 = evaluate.load('f1')
    mf1 = evaluate.load('f1')
    progress_bar = tqdm(range(len(test_dataloader)))

    all_predictions = []
    all_references = []

    model.eval()

    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        all_predictions.extend(predictions.cpu().numpy())
        all_references.extend(batch['labels'].cpu().numpy())
        progress_bar.update(1)

    accuracy = acc.compute(predictions=all_predictions, references=all_references)
    f1_score = f1.compute(predictions=all_predictions, references=all_references, average=None)
    macro_f1 = mf1.compute(predictions=all_predictions, references=all_references, average='macro')

    return accuracy, f1_score, macro_f1

def build_model():
    tokenized_dataset=get_dataset().map(tokenize_function, batched=True)

    tokenized_dataset = tokenized_dataset.remove_columns(['comment_text'])
    tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')
    tokenized_dataset.set_format('torch')


    train_dataloader = DataLoader(tokenized_dataset['train'], batch_size=4, shuffle=True)
    test_dataloader = DataLoader(tokenized_dataset['test'], batch_size=4, shuffle=True)

    num_training_steps = len(train_dataloader) * num_epochs
    scheduler = get_scheduler(name='linear',optimizer=optimizer,num_warmup_steps=50,num_training_steps=num_training_steps)

    train(model, train_dataloader, optimizer, scheduler, num_epochs, num_training_steps)
    acc, f1, mf1 = eval(model, test_dataloader)

    print("Result: Weibo_emoji")
    print('Accuracy: ', acc['accuracy'])
    print('F1_sex: ', f1['f1'][1])
    print('F1_not: ', f1['f1'][0])
    print('F1_macro: ', mf1['f1'])

    result = pd.DataFrame([acc['accuracy'], f1['f1'][1], f1['f1'][0], mf1['f1']], columns=['Weibo_emoji'],
                          index=['Acc', 'F1_sex', 'F1_not', 'F1_macro'])
    result.to_csv('Weibo_emoji.csv')

if __name__ == '__main__':
    build_model()


























