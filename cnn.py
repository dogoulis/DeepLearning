#IMPORTS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.legacy import data
from torchtext.legacy import datasets
import random
import numpy as np
import time
# ======================================================================================================================

# getting data:

text_data = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', batch_first=True)
labels = data.LabelField(dtype=torch.float)

# defining random seed
seed = 42
rd = random.seed(seed)
# ======================================================================================================================
# splitting data into test and train
train_data, test_data = datasets.IMDB.splits(text_data, labels)
# splitting train data into train and validation
train_data, validation_data = train_data.split(random_state=rd)
# setting voc_size:
vocabulary_size = 20000
# building vocabulary
text_data.build_vocab(train_data, max_size=vocabulary_size, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
labels.build_vocab(train_data)
# ======================================================================================================================
# cuda for google Colab
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
# ======================================================================================================================
# defining batch size
batch_size = 64

# creating iterators
train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, validation_data, test_data), batch_size=batch_size, device=device)
# ======================================================================================================================
# creating the text CNN class
class textCNN(nn.Module):
    def __init__(self, vocabulary_size, emb_dim, out_dim, no_filters, filter_sizes, pad_index,
                 dropout):
        
        super().__init__()


        self.embedding = nn.Embedding(vocabulary_size, emb_dim, padding_idx=pad_index)

        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=no_filters, kernel_size=(filter_sizes[0], emb_dim))
        
        self.conv_2 = nn.Conv2d(in_channels=1, out_channels=no_filters, kernel_size=(filter_sizes[1], emb_dim))
        
        self.conv_3 = nn.Conv2d(in_channels=1, out_channels=no_filters, kernel_size=(filter_sizes[2], emb_dim))
        
        self.fc = nn.Linear(len(filter_sizes) * no_filters, out_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):

        # embedding layer:
        embedded = self.embedding(text)
        
        embedded = embedded.unsqueeze(1)

        # conv layers:
        conv_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conv_2 = F.relu(self.conv_2(embedded).squeeze(3))
        conv_3 = F.relu(self.conv_3(embedded).squeeze(3))
        # mac pool layers:
        pool_1 = F.max_pool1d(conv_1, conv_1.shape[2]).squeeze(2)
        pool_2 = F.max_pool1d(conv_2, conv_2.shape[2]).squeeze(2)
        pool_3 = F.max_pool1d(conv_3, conv_3.shape[2]).squeeze(2)

        # dropout layer for regularization
        reg = self.dropout(torch.cat((pool_1, pool_2, pool_3), dim=1))
            
        return self.fc(reg)
# ======================================================================================================================
# setting the parameters:
inputDim = len(text_data.vocab)
embeddingDim = 100
outputDim = 1
noFilters = 150
filterSizes = [3, 4, 5]
padIndex = text_data.vocab.stoi[text_data.pad_token]
dropout = 0.8
# ======================================================================================================================
# defining the model:
model = textCNN(inputDim, embeddingDim, outputDim, noFilters, filterSizes, padIndex, dropout)
# ======================================================================================================================
# calling the pretrained embeddings:
pretrained_embeddings = text_data.vocab.vectors
# setting them into the model:
model.embedding.weight.data.copy_(pretrained_embeddings)
# getting the indexes of <unk> tokens
unk = text_data.vocab.stoi[text_data.unk_token]
# initialize all embeddings to zeroes in order to avoid using <unk> and <pad> tokens.
model.embedding.weight.data[unk] = torch.zeros(embeddingDim)
model.embedding.weight.data[padIndex] = torch.zeros(embeddingDim)
# ======================================================================================================================
# getting the model into the device (GPU)
model = model.to(device)
# defining optimizer:
optimizer = optim.Adam(model.parameters())
# defining loss function
criterion = nn.BCEWithLogitsLoss()
# setting the criterion into the device (GPU)
criterion = criterion.to(device)
# ======================================================================================================================
# defining binary accuracy:


def accuracy(predicted, y):

    roundPredicted = torch.round(torch.sigmoid(predicted))
    D = (roundPredicted == y).float() 
    accuracy = D.sum() / len(D)
    return accuracy

# defining the train function:


def train(model, iterator, optimizer, criterion):
    
    epochLOSS = 0
    epochACC = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.text).squeeze(1)
        
        loss = criterion(predictions, batch.label)
        
        acc = accuracy(predictions, batch.label)
        
        epochLOSS.backward()
        
        optimizer.step()
        
        epochLOSS = epochLOSS + loss.item()
        epochACC = epochACC + acc.item()
        
    return epochLOSS / len(iterator), epochACC / len(iterator)

# defining the evaluation function:

def evaluation(model, iterator, criterion):

    epochLOSS = 0
    epochACC = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = accuracy(predictions, batch.label)

            epochLOSS = epochLOSS + loss.item()
            epochACC = epochACC + acc.item()
        
    return epochLOSS / len(iterator), epochACC / len(iterator)

# defining function for time


def epoch_time(startTime, endTime):
    t = endTime - startTime
    tmins = int(t / 60)
    tsecs = int(t - (tmins * 60))
    return tmins, tsecs

# ======================================================================================================================
# trainning the model

# setting epochs
epochs = 20

# setting an upper bound for valid loss:
best_validation_score = float('inf')

for epoch in range(epochs):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iter, optimizer, criterion)
    validation_loss, validation_acc = evaluation(model, valid_iter, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    # saving model:
    if validation_loss < best_validation_score:
        best_valid_loss = validation_loss
        torch.save(model.state_dict(), 'modelTEXTCNN.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s', f'\tTrain Loss: {train_loss:.2f} | Train Acc: {train_acc*100:.2f}%',
          f'\t Val. Loss: {validation_loss:.3f} |  Val. Acc: {validation_acc*100:.2f}%')
# ======================================================================================================================
# loading model for predictions:
model.load_state_dict(torch.load('modelTEXTCNN.pt'))

# predicting and evaluating:
testLoss, testAccuracy = evaluation(model, test_iter, criterion)

# printing the evaluation:
print(f'The test loss for the CNN model is: {testLoss:.2f} |The test accuracy for the CNN model is: {testAccuracy*100:.2f}%')
# ======================================================================================================================