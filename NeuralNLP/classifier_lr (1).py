from typing import List, Any
from random import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from tqdm.notebook import tqdm
import numpy as np
import wget

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

def get_phrase_embedding(tokens, model):
    """
    Convert phrase (tokens) to a vector by aggregating it's word embeddings. See description above.
    """

    # average word vectors for all words in tokenized phrase
    # skip words that are not in model's vocabulary
    # if all words are missing from vocabulary, return zeros
    
    
    vector = np.zeros([model.vector_size], dtype='float32')

    used_words = 0
    
    for word in tokens:
        if word in model:
            vector += model.wv[word]
            used_words += 1
    
    if used_words > 0:
        vector = vector / used_words
    
    return vector

class LogisticRegression(nn.Module):
    def __init__(self, num_labels, vocab_size):
        super().__init__()
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, vec):
        return F.log_softmax(self.linear(vec), dim=1)

def make_dataloader(train_data, y_enc_train):
    X_train = []
    if y_enc_train is not None:
        labels = torch.from_numpy(y_enc_train).float()
    
        for k, value in enumerate(train_data):
            feature = torch.from_numpy(value)
            X_train.append((feature, labels[k]))
    else:
         for k, value in enumerate(train_data):
            feature = torch.from_numpy(value)
            X_train.append((feature))
    return X_train

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
        BATCH_SIZE, batch_size, N_EPOCHS, LEARNING_RATE = pretrain_params
    # preprocess text
    train_texts = [preprocess(txt) for txt in train_texts] 

    #make vocab
    # fdist = FreqDist()
    # for sent in train_texts:
    #     for word in sent:
    #         fdist[word] += 1
    # # CBOW matrix from numpy 2 torch
    # X_train = to_mtx(train_texts, fdist).tocoo()
    # X_train = torch.sparse.LongTensor(torch.LongTensor([X_train.row.tolist(), X_train.col.tolist()]),
    #                           torch.LongTensor(X_train.data.astype(np.int32)))            
    
    # download embs
    EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin.gz'
    url = 'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz'
    
    if EMBEDDING_FILE not in os.listdir('.'):
        print('Download EMBS!')
        filename = wget.download(url)
    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
    # get document embs by average word embs
    train_word2vec = [get_phrase_embedding(txt, word2vec) for txt in train_texts] 
    # prprocess labels
    y_train = np.array([1 if label == 'pos' else 0 for label in train_labels ])
    
    encoder = OneHotEncoder(categories=[range(2)], sparse=False)
    y_enc_train = encoder.fit_transform(y_train.reshape(-1, 1))

    X_train = make_dataloader(train_word2vec, y_enc_train)
    train_dataloader = DataLoader(X_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    VOCAB_SIZE = 300
    NUM_LABELS = len(set(y_train))

    model = LogisticRegression(NUM_LABELS, VOCAB_SIZE)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.BCEWithLogitsLoss()

    for _ in tqdm(range(N_EPOCHS)):
        for features, labels in train_dataloader:       
            model.zero_grad()
            log_probs = model(features)
            loss = loss_function(log_probs, labels)
            loss.backward()
            optimizer.step()

    params = [model, batch_size, word2vec]
    return params 
    

def pretrain(texts_list: List[List[str]]) -> Any:
    """
    Pretrain classifier on unlabeled texts. If your classifier cannot train on unlabeled data, skip this.
    :param texts_list: a list of list of texts (str objects), one str per example.
        It might be several sets of texts, for example, train and unlabeled sets.
    :return: learnt parameters, or any object you like (it will be passed to the train function)
    """
    # ############################ PUT YOUR CODE HERE #######################################
    BATCH_SIZE = 50
    batch_size = 200
    N_EPOCHS = 10
    LEARNING_RATE = 0.001

    pretrain_params = [ 
            BATCH_SIZE, batch_size, N_EPOCHS, LEARNING_RATE
    ]
    return pretrain_params 

def predict(model,data_loader,device=None):
    y_pred_all = np.array([])
    model.eval()
    for features in data_loader:
        if device is not None:
            features = features.to(device)    
        with torch.no_grad():
            y_test_pred = model(features)

            y_pred_all = np.append(y_pred_all,np.argmax(y_test_pred,axis=1).cpu().detach().numpy())
    return y_pred_all

def classify(texts: List[str], params: Any) -> List[str]:
    """
    Classify texts given previously learnt parameters.
    :param texts: texts to classify
    :param params: parameters received from train function
    :return: list of labels corresponding to the given list of texts
    """
    model = params[0]
    batch_size = params[1]
    word2vec = params[2]

    # preprocess text
    texts = [preprocess(txt) for txt in texts] 
    texts_word2vec = [get_phrase_embedding(txt, word2vec) for txt in texts] 

    X_test = make_dataloader(texts_word2vec, None)
    test_dataloader = DataLoader(X_test, batch_size=batch_size, shuffle=False, num_workers=1)


    res = []
   
    y_pred_all = predict(model,test_dataloader,device=device)
        
    for pred in y_pred_all:
        if pred:
            res.append('pos')
        else:
            res.append('neg')
    return res   
    # ############################ REPLACE THIS WITH YOUR CODE #############################
