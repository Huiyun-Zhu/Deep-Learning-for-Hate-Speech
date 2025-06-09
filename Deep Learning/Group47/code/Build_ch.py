import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm


Acc=[]
F1_sex=[]
F1_not=[]
F1_macro=[]
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large")
id2label = {0: "NONHATE", 1: "HATE"}
label2id = {"NONHATE": 0, "HATE": 1}
num_epochs = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def get_dataset(train_path, test_path):
    dataset = load_dataset('csv', data_files={'train': train_path, 'test': test_path})
    return dataset

def tokenize_function(example):
    return tokenizer(example['text'],padding="max_length",truncation=True,return_tensors="pt")

def token_process(tokenized_dataset):
    tokenized_dataset=tokenized_dataset.remove_columns([col for col in tokenized_dataset['train'].column_names if col not in ['label','input_ids','attention_mask','token_type_ids']])
    tokenized_dataset=tokenized_dataset.rename_column('label','labels')
    tokenized_dataset.set_format('torch')
    return tokenized_dataset

def token_dataloader(tokenized_dataset):
    train_dataloader = DataLoader(tokenized_dataset['train'], batch_size=4, shuffle=True)
    test_dataloader = DataLoader(tokenized_dataset['test'], batch_size=4, shuffle=True)
    return train_dataloader, test_dataloader

def test_token_dataloader(test_path):
    test_dataset = load_dataset('csv', data_files={'test': test_path})
    tokenized_dataset = test_dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns([col for col in tokenized_dataset['test'].column_names if
                                                          col not in ['label', 'input_ids', 'attention_mask',
                                                                      'token_type_ids']])
    tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')
    tokenized_dataset.set_format('torch')
    test_dataloader = DataLoader(tokenized_dataset['test'], batch_size=4, shuffle=True)
    return test_dataloader


def train_step(train_dataloader,optimizer):
    num_training_steps = len(train_dataloader) * num_epochs
    scheduler = get_scheduler(name='linear',optimizer=optimizer,num_warmup_steps=10,num_training_steps=num_training_steps)
    return num_training_steps, scheduler

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

def print_results(n, acc, f1_sex, f1_not, mf1):
    print('Result from round ', n)
    print('Accuracy: ',acc)
    print('F1_sex: ', f1_sex)
    print('F1_not: ', f1_not)
    print('F1_macro: ', mf1)

def save_model(model, model_name):
    model.save_pretrained(model_name)

def load_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    # optimizer = AdamW(model.parameters(), lr=1e-6)
    return model

def plot_result(result):
    plt.figure(figsize=(15,5))
    plt.plot(result.loc['Acc'],label='Accuracy',marker='o',color='r')
    plt.plot(result.loc['F1_macro'],label='F1_Macro',marker='o',color='b',linestyle='--')
    plt.legend(loc='best')
    plt.title('Accuracy & F1_Macro for R1-R4')
    plt.ylabel('Metrics')
    plt.savefig('result_build.png')
    plt.show()

def build_model(model, train_path, test_path, model_name):
    tokenized_dataset = get_dataset(train_path, test_path).map(tokenize_function, batched=True)
    tokenized_dataset = token_process(tokenized_dataset)
    train_dataloader, test_dataloader = token_dataloader(tokenized_dataset)

    optimizer = AdamW(model.parameters(), lr=1e-6)
    num_training_steps, scheduler = train_step(train_dataloader, optimizer)
    model.to(device)

    train(model, train_dataloader, optimizer, scheduler, num_epochs, num_training_steps)
    acc, f1, mf1 = eval(model, test_dataloader)
    save_model(model,model_name)

    return acc, f1, mf1



if __name__ == '__main__':
    model_r1 = AutoModelForSequenceClassification.from_pretrained('Microsoft/deberta-large', num_labels=2,
                                                                  id2label=id2label, label2id=label2id)
    acc,f1,mf1 = build_model(model_r1, 'Emoji_build_ch/train/r1_train.csv', 'Emoji_build_ch/test/r1_test.csv', 'model_r1')
    Acc.append(acc['accuracy'])
    F1_sex.append(f1['f1'][1])
    F1_not.append(f1['f1'][0])
    F1_macro.append(mf1['f1'])
    print_results(1, Acc[0], F1_sex[0], F1_not[0], F1_macro[0])


    model_r2 = load_model('model_r1')
    acc,f1,mf1 = build_model(model_r2, 'Emoji_build_ch/train/r2_train.csv', 'Emoji_build_ch/test/r2_test.csv', 'model_r2')
    Acc.append(acc['accuracy'])
    F1_sex.append(f1['f1'][1])
    F1_not.append(f1['f1'][0])
    F1_macro.append(mf1['f1'])
    print_results(2, Acc[1], F1_sex[1], F1_not[1], F1_macro[1])


    model_r3 = load_model('model_r2')
    acc,f1,mf1 = build_model(model_r3, 'Emoji_build_ch/train/r3_train.csv', 'Emoji_build_ch/test/r3_test.csv', 'model_r3')
    Acc.append(acc['accuracy'])
    F1_sex.append(f1['f1'][1])
    F1_not.append(f1['f1'][0])
    F1_macro.append(mf1['f1'])
    print_results(3, Acc[2], F1_sex[2], F1_not[2], F1_macro[2])


    model_r4 = load_model('model_r3')
    acc,f1,mf1 = build_model(model_r4, 'Emoji_build_ch/train/r4_train.csv', 'Emoji_build_ch/test/r4_test.csv', 'model_r4')
    Acc.append(acc['accuracy'])
    F1_sex.append(f1['f1'][1])
    F1_not.append(f1['f1'][0])
    F1_macro.append(mf1['f1'])
    print_results(4, Acc[3], F1_sex[3], F1_not[3], F1_macro[3])

    result = pd.DataFrame([Acc, F1_sex, F1_not, F1_macro], columns=['R1', 'R2', 'R3', 'R4'],
                          index=['Acc', 'F1_hate', 'F1_not', 'F1_macro'])
    result.to_csv('result_build.csv')
    plot_result(result)






