import os
import numpy as np
import pandas as pd
import torch
import nltk

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, cohen_kappa_score, make_scorer


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


get_ipython().system('pip install transformers')

from transformers import BertTokenizer,BertModel,BertForSequenceClassification,BertPreTrainedModel
from transformers import DistilBertModel,AutoTokenizer, AutoModelForSequenceClassification



inter=pd.read_excel("../mnt/nas2/jaehyun/Inter.xlsx")
inter.head()



dataset = inter[['Essay','Language']]
dataset.dropna(axis=0)

data=dataset['Essay'].tolist()
label=dataset['Language'].tolist()

essay_train, essay_test, score_train, score_test=train_test_split(data, label, test_size=0.2, shuffle=True, random_state=10)
essay_train, essay_valid, score_train, score_valid=train_test_split(essay_train, score_train,test_size=0.2, shuffle=True, random_state=9 )


dataset = inter[['Essay','Content','Language']]
dataset.dropna(axis=0)


#tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')
#model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=1)
tokenizer=BertTokenizer.from_pretrained('bert-base-cased')
model=BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=1)
model.to(device)


train_encoding=tokenizer(essay_train, truncation=True, padding="max_length")
test_encoding=tokenizer(essay_test, truncation=True, padding="max_length")
valid_encoding=tokenizer(essay_valid, truncation=True, padding="max_length" )



class AESDataset(torch.utils.data.Dataset):
  def __init__(self, encodings, labels):
    self.encodings=encodings
    self.labels=labels
  
  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    item={key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    #return self.encodings, self.labels
    return item


train_dataset=AESDataset(train_encoding, score_train)
test_dataset=AESDataset(test_encoding, score_test)
valid_dataset=AESDataset(valid_encoding, score_valid)
#train_dataset[2]



token_cls=tokenizer.cls_token
token_sep=tokenizer.sep_token


from transformers import TrainingArguments, Trainer
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=30,              # total number of training epochs
    per_device_train_batch_size=4,  # batch size per device during training
    per_device_eval_batch_size=4,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    learning_rate=1e-5
)


trainer = Trainer(model=model,args=training_args,train_dataset=train_dataset,
                  eval_dataset=valid_dataset)

trainer.train()



trainer.evaluate()



test_loader=DataLoader(test_dataset, batch_size=4, shuffle=False)
total_results=[]
pred_result=[]
labels_result=[]

for batch in test_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    outputs = model(input_ids, attention_mask=attention_mask)
    #outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    #logits = outputs
    
    pred=outputs.logits.cpu()
    pred=pred.detach().numpy()
    pred=pred*10
    pred=np.around(pred, 0)
    #pred=np.around(pred,1)
    np.nan_to_num(pred)
    pred=np.squeeze(pred,1)
    #print(pred)
    
    labels=labels.to('cpu').numpy()
    #labels=np.around(labels,0)
    labels=labels*10
    #print(labels)
    print(pred, labels)
    result= cohen_kappa_score(labels,pred ,weights='quadratic')
    #print('Test QWK:', result)
    total_results.append(result)


print('Test QWK average:', np.mean(np.asarray(total_results)))
print('Test QWK std:', np.std(np.asarray(total_results)))


