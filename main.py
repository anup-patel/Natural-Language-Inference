



############ Logistic Regression ############
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import json


print("########################")
print("Dept:: CSA")
print("Person Name:: Anup Patel")
print("SR:: 15474")
print("########################")

#load train data
path='../data/snli_1.0/'
train_file=path + 'snli_1.0_train.jsonl'
val_file=path + 'snli_1.0_dev.jsonl'
test_file=path + 'snli_1.0_test.jsonl'


print("Data Preprocessing for linear model \n")
import json
train_data = []
with open(train_file) as f:
    for line in f:
        train_data.append(json.loads(line))

import json
val_data = []
with open(val_file) as f:
    for line in f:
        val_data.append(json.loads(line))


import json
test_data = []
with open(test_file) as f:
    for line in f:
        test_data.append(json.loads(line))

train_data=pd.DataFrame(train_data)
val_data=pd.DataFrame(val_data)
test_data=pd.DataFrame(test_data)


#Train data preprocess
train_data = train_data.dropna(subset = ['sentence2'])
train_data = train_data.dropna(subset = ['sentence1'])
train_data = train_data[train_data["gold_label"] != "-"]


#Validation data preprocess
val_data = val_data.dropna(subset = ['sentence2'])
val_data = val_data.dropna(subset = ['sentence1'])
val_data = val_data[val_data["gold_label"] != "-"]

#Test data preprocess
# test_data = test_data.dropna(subset = ['sentence2'])
# test_data = test_data.dropna(subset = ['sentence1'])
# test_data = test_data[test_data["gold_label"] != "-"]

# len(test_data)

### Data Preprocessing

train_data = train_data.applymap(lambda s:s.lower() if type(s) == str else s)
# train_data

val_data = val_data.applymap(lambda s:s.lower() if type(s) == str else s)
test_data = test_data.applymap(lambda s:s.lower() if type(s) == str else s)
train_features=train_data[['sentence1','sentence2']]
val_features=val_data[['sentence1','sentence2']]
test_features=test_data[['sentence1','sentence2']]

# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# stop=set(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
#tfidf = TfidfVectorizer(ngram_range=(1,5), max_features=100,stop_words=stop)
# tfidf = TfidfVectorizer(stop_words=stop)
tfidf = TfidfVectorizer()
## Training data
train_sentence=train_features['sentence1']+train_features['sentence2']
train_features=tfidf.fit_transform(train_sentence)
## Validation data
val_sentence=val_features['sentence1']+val_features['sentence2']
val_features=tfidf.transform(val_sentence)
## Testing data
test_sentence=test_features['sentence1']+test_features['sentence2']
test_features=tfidf.transform(test_sentence)
### Train label
train_label=train_data['gold_label']
train_label=train_label.replace(to_replace ="neutral",value =0)
train_label=train_label.replace(to_replace ="entailment",value =1)
train_label=train_label.replace(to_replace ="contradiction",value =2)
train_label=(train_label.values).astype(int)

### Validation label
val_label=val_data['gold_label']
val_label=val_label.replace(to_replace ="neutral", value =0)
val_label=val_label.replace(to_replace ="entailment", value =1)
val_label=val_label.replace(to_replace ="contradiction",value =2)
val_label=(val_label.values).astype(int)

### Test label
test_label=test_data['gold_label']
test_label=test_label.replace(to_replace ="neutral",value =0)
test_label=test_label.replace(to_replace ="entailment", value =1)
test_label=test_label.replace(to_replace ="contradiction",value =2)
test_label=test_label.replace(to_replace ="-",value =2)
test_label=(test_label.values).astype(int)

print("Data Preprocessing Completed \n")

#### Linear Regression Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# clf = LogisticRegression(solver = 'lbfgs',max_iter=100).fit(X_train, y_train)
# clf = LogisticRegression().fit(train_features, train_label)


# # save the model to disk
filename = 'model/logistic_model.sav'
# pickle.dump(clf, open(filename, 'wb'))

# load the model from disk
clf = pickle.load(open(filename, 'rb'))

## Validation
prediction=clf.predict(val_features)
# print(" Validation Accuracy :: " , round(accuracy_score(prediction, val_label)*100) , "%")
# Validation Accuracy ::  55.0 % (removing stop word)

# Testing
prediction=clf.predict(test_features)
# print("Test Accuracy :: " , round(accuracy_score(prediction, test_label)*100) , "%")
# Test Accuracy ::  55.0 % ((removing stop word))

file=open("tfidf.txt","w+")
# file.write("Accuracy on Test Data : 57 % \n")
# file.write("actual_label, predicted_label \n")
for i in range(len(prediction)):    
    if(prediction[i]==0):
        file.write('neutral \n')
    if(prediction[i]==1):
        file.write("entailment \n")
    if(prediction[i]==2 or prediction[i]=='-'):
        file.write("contradiction \n") 
file.close()
print("tfidf.txt file Generated \n")



########################## Neural Network Model #########

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import torch.optim as O
import torch.nn as nn
from torchtext import data
import torchtext
from torchtext import datasets
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device='cpu'
# print(device)


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext import data
from torchtext import datasets
import random

print("Data Preprocessing for Neural Network ... \n")
inputs = data.Field(lower=True,tokenize='spacy', batch_first = True)
answers = data.Field(sequential=False, unk_token = None, is_target = True)
train, dev, test = datasets.SNLI.splits(inputs, answers)


# print(train[0].premise)
# print(train[0].hypothesis)
# inputs.build_vocab(train, dev,vectors="glove.6B.100d")
inputs.build_vocab(train, dev)
answers.build_vocab(train)
label=answers.vocab.stoi

train_iter = data.Iterator((train),batch_size=64,device=device,train=True)
test_iter = data.Iterator((test),batch_size=64,device=device,shuffle=False)

dev_iter = data.Iterator((dev),batch_size=64,device=device,shuffle=False)

# print ("Length of Text Vocabulary: " + str(len(inputs.vocab)))
# # print ("Vector size of Text Vocabulary: ", inputs.vocab.vectors.size())
# print ("Label Length: " + str(len(answers.vocab)))

n_embed = len(inputs.vocab)
d_out = len(answers.vocab)
n_cells = 2

print("Data Preprocessing for Neural Network Completed \n")
#LSTM Model
class BiLSTM(nn.Module):
    def __init__(self, vocab_size,embed_dim,dp_ratio,d_hidden,out_dim):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.projection = nn.Linear(embed_dim, 300)
        self.dropout = nn.Dropout(p = dp_ratio)
        self.lstm = nn.LSTM(300, d_hidden, 3)
        self.relu = nn.ReLU()
        self.out = nn.Sequential(
            nn.Linear(1024, 1024),
            self.relu,
            self.dropout,
            nn.Linear(1024, 1024),
            self.relu,
            self.dropout,
            nn.Linear(1024, 1024),
            self.relu,
            self.dropout,
            nn.Linear(1024, out_dim)
        )
        pass

    def forward(self, batch):
        premise_embed = self.embedding(batch.premise)
        hypothesis_embed = self.embedding(batch.hypothesis)
        premise_proj = self.relu(self.projection(premise_embed))
        hypothesis_proj = self.relu(self.projection(hypothesis_embed))
        encoded_premise, _ = self.lstm(premise_proj)
        encoded_hypothesis, _ = self.lstm(hypothesis_proj)
        premise = encoded_premise.sum(dim = 1)
        hypothesis = encoded_hypothesis.sum(dim = 1)
        combined = torch.cat((premise, hypothesis), 1)
        return self.out(combined)

import torch.optim as O
# learning_rate = 0.001
# batch_size = 32
# output_size = 2
# hidden_size = 256
# embedding_length = 300
vocab_size=len(inputs.vocab)
embed_dim=300
dp_ratio=0.01
d_hidden=512
out_dim=3
model = BiLSTM(vocab_size,embed_dim,dp_ratio,d_hidden,out_dim).to(device)
loss_fn = F.cross_entropy
# opt=O.Adam(model.parameters(), lr = learning_rate) # with learning rate
opt=O.Adam(model.parameters()) # without learning rate
criterion = nn.CrossEntropyLoss(reduction = 'sum')



import torch
import torch.nn as nn


def train_model(model, train_iter, epoch):
    model.to(device)
    model.train()
    n_correct, n_total, n_loss = 0, 0, 0
    for batch_idx, batch in enumerate(train_iter):
        
        # batch=batch.to(device)
        opt.zero_grad()
        answer = model(batch)
        loss = criterion(answer, batch.label)
        
        n_correct += (torch.max(answer, 1)[1].view(batch.label.size()) == batch.label).sum().item()
        n_total += batch.batch_size
        n_loss += loss.item()
        
        loss.backward(); opt.step()
    train_loss = n_loss/n_total
    train_acc = 100. * n_correct/n_total
    return train_loss, train_acc

def val_model(model, val_iter):
    model.to(device)
    model.eval()
    n_correct, n_total, n_loss = 0, 0, 0
    for batch_idx, batch in enumerate(val_iter):
        # print(batch_idx)
        answer = model(batch)
        loss = criterion(answer, batch.label)
        n_correct += (torch.max(answer, 1)[1].view(batch.label.size()) == batch.label).sum().item()
        n_total += batch.batch_size
        n_loss += loss.item()
    val_loss = n_loss/n_total
    val_acc = 100. * n_correct/n_total
    return val_loss, val_acc




# max_acc=0
# train_loss_array=[]
# val_loss_array=[]
# train_acc_array=[]
# val_acc_array=[]
# for epoch in range(50):
#     train_loss, train_acc = train_model(model, train_iter, epoch)
#     val_loss, val_acc = val_model(model, dev_iter)
#     train_loss_array.append(train_loss)
#     val_loss_array.append(val_loss)
#     train_acc_array.append(train_acc)
#     val_acc_array.append(val_acc)

#     print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')
#     if(val_acc>max_acc):
#         max_acc=val_acc
#     ##Save Model
#     torch.save(model,'model/casl_gpu_pytorch_lstm_model_iter'+str(epoch+1)+'.pth')
#     if(epoch>20 and val_acc<max_acc):
#         break
# f=open("train_val.txt","w+")
# f.write("Train loss \n")
# for item in train_loss_array:
#     f.write("%s " % item)
# f.write("\n Validate loss \n")
# for item in val_loss_array:
#     f.write("%s " % item)
    
# f.write("Train Accuracy \n")
# for item in train_acc_array:
#     f.write("%s " % item)
# f.write("\n Validate Accuracy \n")
# for item in val_acc_array:
#     f.write("%s " % item)
    
# f.close()

##Save Model
# torch.save(model,'/content/drive/My Drive/Colab Notebooks/Project3/torch_model_bilstm.pth')



#Loading Model
# model=torch.load('clserv_pytorch_lstm_model.pth') # To Run on GPU
model=torch.load('model/lstm_model.pth') # To Run on GPU
#model=torch.load('model/lstm_model_new.pth',map_location='cpu') #To Run on CPU
## Train Loss: 0.450, Train Acc: 82.33%, Val. Loss: 0.575763, Val. Acc: 78.50%
### We are using results after training model
train_loss=0.45
train_acc=82.33
val_loss=0.575
val_acc=78.5
print("Results :: ")
print(f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%')
print(f'Val Loss: {val_loss:.3f}, val Acc: {val_acc:.2f}%')
# print("Evaluation Result ::")
#Testing
test_loss, test_acc = val_model(model, test_iter)
print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')


actual=[]
for data in test:
#     print(data.label)
    actual.append(data.label)
    
    
prediction=[]
for batch_idx, batch in enumerate(test_iter):
    # print(batch_idx)
    answer = model(batch)
    for item in torch.max(answer, 1)[1].cpu():
#         print(np.array(item))
        prediction.append(np.array(item))
    
    
file=open("deep_model.txt","w+")
file.write("Test Loss : .55 \n")
file.write("Accuracy on Test Data : 78.90 % \n")
file.write("actual_label,predicted_label \n")
for i in range(len(prediction)):
    file.write(str(actual[i])+',')
    if(prediction[i]==2):
        file.write('neutral\n')
    if(prediction[i]==0):
        file.write("entailment\n")
    if(prediction[i]==1 or prediction[i]=='-'):
        file.write("contradiction\n")
        
    
file.close()



#     print(torch.max(answer, 1)[1])



print("deep_model.txt file Generated \n")