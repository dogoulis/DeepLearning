# IMPORTS
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data
from torchtext.legacy import datasets
import random
import time

# getting data:
# include_lengths returs a tuple [sentence, length of sentence]
text_data = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', include_lengths=True)
labels = data.LabelField(dtype=torch.float)

# defining random seed
seed = 42
rd=random.seed(seed)

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
batch_size=64
# creating iterators
train_iter, validation_iter, test_iter = data.BucketIterator.splits((train_data, validation_data, test_data), batch_size=batch_size,
    sort_within_batch=True,
    device=device)
# ======================================================================================================================
# creating the text RNN (LSTM) class


class RNN(nn.Module):
    def __init__(self, vocabulary_size, emb_dim, hid_dim, out_dim, no_layers, pad_index, dropout,  bidirectional):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocabulary_size, emb_dim, padding_idx=pad_index)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=no_layers, bidirectional=bidirectional, dropout=dropout)
        
        self.fc = nn.Linear(hid_dim * 2, out_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):


     # embedding layer

        embedded = self.dropout(self.embedding(text))

     # packed sentences into device

        packed = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'))
        
        packedOut, (hidden, cell) = self.rnn(packed)

        # unpacking the sentences
        output, outputLengths = nn.utils.rnn.pad_packed_sequence(packedOut)

        # concat the forward and backward LSTM from the second hiden layer and applying dropout
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        return self.fc(hidden)
# ======================================================================================================================
# setting the parameters:
inputDim = len(text_data.vocab)
embeddingDim = 100
hiddenDim = 256
outputDim = 1
no_layers = 2
pad_index = text_data.vocab.stoi[text_data.pad_token]
dropout = 0.8
bidirectional = True
# ======================================================================================================================
# defining the model:
model = RNN(inputDim, embeddingDim, hiddenDim, outputDim, no_layers, pad_index, dropout, bidirectional)
# ======================================================================================================================
# calling the pretrained embeddings:
pretrained_embeddings = text_data.vocab.vectors
# setting them into the model:
model.embedding.weight.data.copy_(pretrained_embeddings)
# getting the indexes of <unk> tokens
unk = text_data.vocab.stoi[text_data.unk_token]
# initialize all embeddings to zeroes in order to avoid using <unk> and <pad> tokens.
model.embedding.weight.data[unk] = torch.zeros(embeddingDim)
model.embedding.weight.data[pad_index] = torch.zeros(embeddingDim)
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
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        text, text_lengths = batch.text
        
        predictions = model(text, text_lengths).squeeze(1)
        
        loss = criterion(predictions, batch.label)
        
        acc = accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# defining the evaluation function:


def evaluation(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            text, text_lengths = batch.text
            
            predictions = model(text, text_lengths).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

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
        torch.save(model.state_dict(), 'modelTEXTLSTM.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s',
          f'\tTrain Loss: {train_loss:.2f} | Train Acc: {train_acc * 100:.2f}%',
          f'\t Val. Loss: {validation_loss:.3f} |  Val. Acc: {validation_acc * 100:.2f}%')

# ======================================================================================================================
# loading model for predictions:
model.load_state_dict(torch.load('modelTEXTLSTM.pt'))
# predicting and evaluating:
testLoss, testAccuracy = evaluate(model, test_iter, criterion)
# printing the evaluation:
print(f'The test loss for the LSTM model is: {testLoss:.3f} |The test accuracy for the LSTM model is: {testAccuracy*100:.2f}%')
# ======================================================================================================================