# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:31:38 2019

@author: Maxen
"""


import io
import numpy as np
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

#word embeddings : anglais et français

def load_vec(emb_path, nmax=50000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)                                                                                                                    
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id

src_path = r'C:\Users\Maxen\Downloads\ok\ok\wiki.multi.fr.vec'
tgt_path = r'C:\Users\Maxen\Downloads\ok\ok\wiki.multi.en.vec'
nmax = 50000  # maximum number of word embeddings to load

src_embeddings, src_id2word, src_word2id = load_vec(src_path, nmax)
tgt_embeddings, tgt_id2word, tgt_word2id = load_vec(tgt_path, nmax)


#données : train et test

def load_dic(path):
    dico_full = {}
    vectors_src=[]
    vectors_tgt = []
    with io.open(path,'r',encoding='utf_8') as f:
        for i,line in enumerate(f):
            word_src, word_tgt = line.rstrip().split(' ',1)
            if word_tgt in tgt_word2id :
                dico_full[word_src]=word_tgt
    for key in dico_full.keys() :
            vectors_src.append(src_embeddings[src_word2id[key]])
            vectors_tgt.append(tgt_embeddings[tgt_word2id[dico_full[key]]])
    X = np.vstack(vectors_src)
    Z = np.vstack (vectors_tgt)
    return dico_full,X,Z

path_train = r'C:\Users\Maxen\Downloads\ok\ok\fr-en.0-5000.txt'
path_test = r'C:\Users\Maxen\Downloads\ok\ok\fr-en.5000-6500.txt'

dico_train, X_train, Z_train = load_dic(path_train)
dico_test, X_test, Z_test = load_dic(path_test)

# LINEAR TRANSFORM : Translation Matrix W 

def C(W,X,Z):
    S=0
    for i in range(X.shape[0]):
        S=S+np.linalg.norm(np.dot(W,X[i])-Z[i])**2
    return S

def dC_dW(W,X,Z):
    S=0
    for i in range(X.shape[0]):
        S=S+2*np.outer((np.dot(W,X[i])-Z[i]),X[i])
    return S

#W = np.random.rand(300,300)
W = np.eye(300)
eta = 0.001
delta = 0.01
N = 300
nb = 1000

#reprendre code theo 

#descente de gradient  
#def gradientDescent(eta,):
valeur_C = []
for t in range(N):
    print(t)
    tmp_W = W 
    W = tmp_W - eta*dC_dW(tmp_W,X_train,Z_train)
    valeur_C.append(C(W,X_train,Z_train))
print(valeur_C)
print(dC_dW(W,X_train,Z_train))
print(np.linalg.norm(dC_dW(W,X_train,Z_train)))

#ou descente de gradient stochastique

norm_grad = []
for t in range(N):
    print(t)
    l = np.random.randint(low=0,high=len(dico_train)) 
    tmp_W = W 
    W = W - eta*dC_dW(tmp_W,X_train,Z_train)
    valeur_C.append(C(W,X_train,Z_train))
print(valeur_C)
print(dC_dW(W,X_train,Z_train))
print(np.linalg.norm(dC_dW(W,X_train,Z_train)))


#ORTHOGONAL TRANSFORM 


#validation : sur dico_test, X_test et Z_test
        
def prediction(W,mot):
    x = src_embeddings[src_word2id[mot]]
    z = np.dot(W,x)
    z_pred = np.argmax(sklearn.metrics.pairwise.cosine_similarity(z.reshape(1,300),tgt_embeddings)) #celui qui a la plus grande similarite cos avec z
    return tgt_id2word[z_pred]

dico_pred={}
i=0
for mot in dico_test.keys() :
    print(i)
    dico_pred[mot]=prediction(W,mot)
    i+=1

def accuracy(dico_pred,dico_test):
    c=0
    for key in dico_test.keys():
        if dico_test[key] == dico_pred[key]:
            c+=1
    return(c/len(dico_test)) #nb de mots bien prédits/nb de mots total
    
accuracy_test = accuracy(dico_pred,dico_test)
