from typing import List, Any
from random import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from tqdm import tqdm
import numpy as np
import wget
import zipfile

import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.probability import FreqDist
# from torchtext.vocab import vocab

from scipy.sparse import csr_matrix, hstack, issparse, coo_matrix
from gensim.models import KeyedVectors
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader

torch.manual_seed(1)

wordnet_lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'[a-z]+')
stop_words = set(stopwords.words('english'))


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def preprocess(document):
    """
    TODO: write your preprocessing function, including following steps:
    - convert the whole text to the lowercase;
    - tokenize the text;
    - remove stopwords;
    - lemmatize the text.
    Return: string, resulted list of tokens joined with the space.
    """

    # Convert to lowercase
    document = document.lower()
    # Tokenize
    words = tokenizer.tokenize(document)
    # Removing stopwords
    words = [word for word in words if not word in stop_words]
    # Lemmatizing
    for pos in [wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV]:
        words = [wordnet_lemmatizer.lemmatize(x, pos) for x in words]
    
    #return " ".join(words)
    return words

def to_matrix(lines, vocab, max_len=None, dtype='int32'):
    """Casts a list of lines into a matrix"""
    pad = vocab['EOS']
    max_len = max_len or max(map(len, lines))
    lines_ix = np.zeros([len(lines), max_len], dtype) + pad
    for i in range(len(lines)):
        line_ix = [vocab.get(l, vocab['UNK']) for l in lines[i]]
        lines_ix[i, :len(line_ix)] = line_ix
    lines_ix = torch.LongTensor(lines_ix)
    return lines_ix

def load_embeddings(emb_path, vocab):
    clf_embeddings = {}
    emb_vocab = set()
    for line in open(emb_path):
        line = line.strip('\n').split()
        word, emb = line[0], line[1:]
        emb = [float(e) for e in emb]
        if word in vocab:
            clf_embeddings[word] = emb
    for w in vocab:
        if w in clf_embeddings:
            emb_vocab.add(w)
    word2idx = {w: idx for (idx, w) in enumerate(emb_vocab)}
    max_val = max(word2idx.values())
    
    word2idx['UNK'] = max_val + 1
    word2idx['EOS'] = max_val + 2
    emb_dim = len(list(clf_embeddings.values())[0])
    clf_embeddings['UNK'] = [0.0 for i in range(emb_dim)]
    clf_embeddings['EOS'] = [0.0 for i in range(emb_dim)]
    
    embeddings = [[] for i in range(len(word2idx))]
    for w in word2idx:
        embeddings[word2idx[w]] = clf_embeddings[w]
    embeddings = torch.Tensor(embeddings)
    return embeddings, word2idx

def to_matrix(lines, vocab, max_len=None, dtype='int32'):
    """Casts a list of lines into a matrix"""
    pad = vocab['EOS']
    max_len = max_len or max(map(len, lines))
    lines_ix = np.zeros([len(lines), max_len], dtype) + pad
    for i in range(len(lines)):
        line_ix = [vocab.get(l, vocab['UNK']) for l in lines[i]]
        lines_ix[i, :len(line_ix)] = line_ix
    lines_ix = torch.LongTensor(lines_ix)
    return lines_ix

def generate_data(train_tok,vocab,label_enc=None,with_label=True):
    data = []
    if with_label:
        for t, l in zip(train_tok,label_enc):
            t = to_matrix([t], vocab)
            l = torch.Tensor([l])
            data.append((t, l))
    else:
        for t in train_tok:
            t = to_matrix([t], vocab)
            data.append(t)
    return data

class BiLSTM(nn.Module):
    def __init__(self, embeddings, hidden_dim=128, lstm_layer=1, output=2):
        
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        
        # load pre-trained embeddings
        self.embedding = nn.Embedding.from_pretrained(embeddings)
        # embeddings are not fine-tuned
        self.embedding.weight.requires_grad = False
        
        # RNN layer with LSTM cells
        # OR self.lstm = NaiveLSTM(input_sz = self.embedding.embedding_dim, hidden_sz = hidden_dim)
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer, 
                            bidirectional=True)
        # dense layer
        self.output = nn.Linear(hidden_dim*2, output)
    
    def forward(self, sents):
        x = self.embedding(sents)
        
        # the original dimensions of torch LSTM's output are: (seq_len, batch, num_directions * hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # reshape to get the tensor of dimensions (seq_len, batch, num_directions, hidden_size)
        lstm_out = lstm_out.view(x.shape[0], -1, 2, self.hidden_dim)#.squeeze(1)
        
        # lstm_out[:, :, 0, :] -- output of the forward LSTM
        # lstm_out[:, :, 1, :] -- output of the backward LSTM
        # we take the last hidden state of the forward LSTM and the first hidden state of the backward LSTM
        dense_input = torch.cat((lstm_out[-1, :, 0, :], lstm_out[0, :, 1, :]), dim=1)
        
        y = self.output(dense_input).view([1, 2])
        return y

def binary_accuracy(preds, y):
    # y is either [0, 1] or [1, 0]
    # get the class (0 or 1)
    y = torch.argmax(y, dim=1)
    
    # get the predicted class
    preds = torch.argmax(torch.sigmoid(preds), dim=1)
    
    correct = (preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc

def train_epoch(model, train_data, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    # set the model to the training mode
    model.train(mode=True)
    
    for t, l in train_data:
        # reshape the data to n_words x batch_size (here batch_size=1)
        t = t.view((-1, 1))
        # transfer the data to GPU to make it accessible for the model and the loss
        t = t.to(device)
        l = l.to(device)
        
        # set all gradients to zero
        optimizer.zero_grad()
        
        # forward pass of training
        # compute predictions with current parameters
        predictions = model(t)
        # compute the loss
        loss = criterion(predictions, l)
        # compute the accuracy (this is only for report)
        acc = binary_accuracy(predictions, l)
        
        # backward pass (fully handled by pytorch)
        loss.backward()
        # update all parameters according to their gradients
        optimizer.step()
        
        # data for report
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(train_data), epoch_acc / len(train_data)

def predict(model, test_data):
    model.eval()

    y_pred_all = np.array([])

    for t in test_data:
        t = t.view((-1, 1))
        t = t.to(device)

        predictions = model(t)

        y_pred_all =  np.append(y_pred_all,np.argmax(predictions.cpu().detach().numpy(),axis=1))
    
    return y_pred_all

def train(
        train_texts: List[str],
        train_labels: List[str],
        pretrain_params: Any = None) -> Any:
    """
    Trains classifier on the given train set represented as parallel lists of texts and corresponding labels.
    :param train_texts: a list of texts (str objects), one str per example
    :param train_labels: a list of labels, one label per example
    :param pretrain_params: parameters that were learned at the pretrain step
    :return: learnt parameters, or any object you like (it will be passed to the classify function)
    """
    if pretrain_params is not None:
        N_EPOCHS, LEARNING_RATE, hidden_dim, layers = pretrain_params
    # preprocess text
    train_texts = [preprocess(txt) for txt in train_texts] 

    #make vocab
    fdist = FreqDist()
    for sent in train_texts:
        for word in sent:
            fdist[word] += 1
    vocab = set(fdist.keys())

    # download embs
    EMBEDDING_FILE = 'glove.6B.300d.txt'
    url = 'http://nlp.stanford.edu/data/glove.6B.zip'
    
    if EMBEDDING_FILE not in os.listdir('.'):
        print('Download EMBS!')
        filename = wget.download(url)
        directory_to_extract_to ='.'
        print('Unzip EMBS!')
        with zipfile.ZipFile('glove.6B.zip', 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
    # word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
    
    embeddings, vocab = load_embeddings(EMBEDDING_FILE, vocab)

    # get document embs by average word embs
    # train_word2vec = [get_phrase_embedding(txt, word2vec) for txt in train_texts] 

    # prprocess labels
    y_train = np.array([1 if label == 'pos' else 0 for label in train_labels ])
    
    encoder = OneHotEncoder(categories=[range(2)], sparse=False)
    y_enc_train = encoder.fit_transform(y_train.reshape(-1, 1))
    
    # use dataloader 
    train_x = generate_data(train_tok=train_texts,label_enc=y_enc_train, vocab=vocab)

    #Initialise the model, optimiser, and loss:
    # hidden_dim = 128
    # layers = 1
    model = BiLSTM(embeddings, hidden_dim, lstm_layer=layers)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    #Transfer the model and loss to GPU:
    model = model.to(device)
    criterion = criterion.to(device)
    
    model.train()

    for i in tqdm(range(N_EPOCHS)):
        # print('epoch =',i)
        # set the model to the training mode
        model.train(mode=True)
        
        for t, l in train_x:
            # reshape the data to n_words x batch_size (here batch_size=1)
            t = t.view((-1, 1))
            # transfer the data to GPU to make it accessible for the model and the loss
            t = t.to(device)
            l = l.to(device)
            
            # set all gradients to zero
            optimizer.zero_grad()
            
            # forward pass of training
            # compute predictions with current parameters
            predictions = model(t)
            # compute the loss
            loss = criterion(predictions, l)
            
            # backward pass (fully handled by pytorch)
            loss.backward()
            # update all parameters according to their gradients
            optimizer.step()
            

    params = [model, embeddings, vocab]
    return params 
    

def pretrain(texts_list: List[List[str]]) -> Any:
    """
    Pretrain classifier on unlabeled texts. If your classifier cannot train on unlabeled data, skip this.
    :param texts_list: a list of list of texts (str objects), one str per example.
        It might be several sets of texts, for example, train and unlabeled sets.
    :return: learnt parameters, or any object you like (it will be passed to the train function)
    """
    # ############################ PUT YOUR CODE HERE #######################################
    N_EPOCHS = 8
    LEARNING_RATE = 0.001
    hidden_dim = 128
    layers = 1

    pretrain_params = [ 
            N_EPOCHS, LEARNING_RATE, hidden_dim, layers 
    ]
    return pretrain_params 


def classify(texts: List[str], params: Any) -> List[str]:
    """
    Classify texts given previously learnt parameters.
    :param texts: texts to classify
    :param params: parameters received from train function
    :return: list of labels corresponding to the given list of texts
    """
    model = params[0]
    embeddings = params[1]
    vocab = params[2]

    # preprocess text
    texts = [preprocess(txt) for txt in texts] 
    # texts_word2vec = [get_phrase_embedding(txt, word2vec) for txt in texts] 
    # use dataloader 
    test_x = generate_data(train_tok=texts,label_enc=None, vocab=vocab, with_label=False)

    res = []
   
    y_pred_all = predict(model,test_x)
        
    for pred in y_pred_all:
        if pred:
            res.append('pos')
        else:
            res.append('neg')
    return res   
    # ############################ REPLACE THIS WITH YOUR CODE #############################
