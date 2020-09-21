from gensim.models import KeyedVectors
import gensim.models.doc2vec as d2v
from gensim.models.doc2vec import TaggedDocument,Doc2Vec

from gensim.test.utils import get_tmpfile

from gensim.models.fasttext import FastText

import networkx as nx
import numpy as np
import gc
import gensim

stopwords = ['rt','amp','url','https','sir','day','title','shri','crore','time',"a", "about","above", "across", "after", "afterwards", "again", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]
tstop = ['rt','amp','url']

import pickle

f = open('data/semeval13_keywords.pkl', 'rb') 
keywords=pickle.load(f)
f.close()

wv_model = FastText.load('model/semeval13_multiplex_walk_fastext.model')

def get_weight_matrix(words,mx_model):
    # define weight matrix dimensions with all 0
    idx=1
    word2idx={}
    vectors=[]
    vectors.append(np.asarray([0]*mx_model.vector_size))
    for i in words:
        word2idx[i]=idx#print(i,len(w2v[i])) i = model.vocab[word].index
        vectors.append(np.asarray(mx_model[i]))
        idx+=1
    return word2idx,vectors


def get_matrix(words,mx_model,SHE_model,emb):
    # define weight matrix dimensions with all 0
    idx=1
    word2idx={}
    vectors=[]
    vectors.append(np.asarray([0]*emb))
    for i in words:
        word2idx[i]=idx#print(i,len(w2v[i]))
        x=mx_model[i]
        x=SHE_model.predict(np.asarray([x]))
        vectors.append(x[0][0])
    return word2idx,vectors

def get_EMB_matrix(words,mx_model,SHE_model,emb):
    # define weight matrix dimensions with all 0
    idx=1
    word2idx={}
    vectors=[]
    vectors.append(np.asarray([0]*emb))
    for i in words:
        word2idx[i]=idx#print(i,len(w2v[i]))
        x=mx_model[i]
        x=SHE_model.predict(np.asarray([x]))
        vectors.append(x[0][0])
    return word2idx,vectors

word2idx,vectors = get_weight_matrix(keywords,wv_model)

from keras.models import model_from_json

vectors=np.array(vectors) 

def get_sentiment(i):
    if i[0] > i[1]:
        if i[0] > i[2]:
            sent='1'
        else:
            sent='0'
    elif i[1] > i[2]:
        sent='-1'
    else:
        sent='0'
    return sent

def convert_labels(labels):
    y_train=labels
    for i in range(len(y_train)):
        if y_train[i]=='1':
            y_train[i]=[1,0,0]
        elif y_train[i]=='-1':
            print(y_train[i])
            y_train[i]=[0,1,0]
        elif y_train[i]=='0':
            y_train[i]=[0,0,1]
    return y_train


import keras
from keras.preprocessing import sequence
from keras.layers import Input, Dense,RepeatVector
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional,SimpleRNN,GRU,CuDNNGRU,Reshape
from keras.layers import Conv1D, GlobalMaxPooling1D, Activation, Flatten, UpSampling1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import plot_model
import tensorflow as tf
from keras.models import model_from_json
import pickle as pkl
from sklearn.metrics import accuracy_score
from random import shuffle
from keras.layers import Concatenate,Multiply,Add
from keras.layers import TimeDistributed
from keras.models import Sequential
import os
from keras.layers import Flatten
from keras.layers import Lambda
from keras import backend as K

def timedistributed_sum(x):
   return K.sum(x, axis=1)

lstmnode=64

def BiLSTM_self_attention_CNN(input_node,max_len):
    encode=Embedding(len(vectors), emb,weights=[vectors], input_length=max_len, trainable=True)(input_node) #,weights=[vectors],trainable=False
    #embed_class_tweets=Reshape((1,lstmnode*2))(embed_tweets)

    encoder_LSTM,_,_,_,_=Bidirectional(LSTM(lstmnode, return_sequences=True, return_state=True, activation='tanh'))(encode)
    attention = TimeDistributed(Dense(input_tweet, activation='softmax'))(encoder_LSTM)
    


    embed_tweets=keras.layers.dot([attention,encoder_LSTM],(1,1), normalize=False)
    embed_class_tweets=Conv1D(emb,
                     1,
                     padding='same',
                     activation='relu',
                     strides=1)(embed_tweets)
    embed_class_tweets=Dropout(0.2)(embed_class_tweets)
    embed_class_tweets=GlobalMaxPooling1D()(embed_class_tweets)
    return embed_class_tweets


def BiLSTM_CNN(input_node,max_len):
    encode=Embedding(len(vectors), emb,weights=[vectors], input_length=max_len, trainable=True)(input_node) #,weights=[vectors],trainable=False
    encoder_LSTM,_,_,_,_=Bidirectional(LSTM(lstmnode, return_sequences=True, return_state=True, activation='tanh'))(encode)
    embed_class_tweets=Conv1D(emb,
                     1,
                     padding='same',
                     activation='relu',
                     strides=1)(encoder_LSTM)
    embed_class_tweets=Dropout(0.2)(embed_class_tweets)
    embed_class_tweets=GlobalMaxPooling1D()(embed_class_tweets)
    return embed_class_tweets


def CNN(input_node,max_len):
    encode_hashtags=Embedding(len(vectors), emb,weights=[vectors], input_length=max_len, trainable=True)(input_node)    #,weights=[vectors],trainable=False
    encode_hashtags=Conv1D(128,
                         3,
                         padding='same',
                         activation='relu',
                         strides=1)(encode_hashtags)
    encode_hashtags=MaxPooling1D(pool_size=1)(encode_hashtags)
    encode_hashtags=Conv1D(64,
                         3,
                         padding='same',
                         activation='relu',
                         strides=1)(encode_hashtags)
    encode_hashtags=Dropout(0.2)(encode_hashtags)

    encode_hashtags=GlobalMaxPooling1D()(encode_hashtags)
    return encode_hashtags

def BiLSTM(input_node,max_len):
    encode=Embedding(len(vectors), emb,weights=[vectors], input_length=max_len, trainable=True)(input_node) #,weights=[vectors],trainable=False
    encoder_BiLSTM=Bidirectional(LSTM(lstmnode,activation='tanh'))(encode)
    return encoder_BiLSTM

def convert2id(text,word2idx=word2idx):
    ids=[]
    for word in text:
        if word in word2idx:
            ids.append(word2idx[word])
    return ids



f_tweet = open('data/semeval13_multiplex_network.pkl', 'rb')
tweet_net=pickle.load(f_tweet)     #((hashtags,mentions,edges,h_k,m_k),res,label)
f_tweet.close()


f_tweet = open('data/semeval13_biased_rw_startnode_without_expansion_lm.pkl', 'rb')  
preprocess_corpus_startnode_id=pickle.load(f_tweet)     #((hashtags,mentions,edges,h_k,m_k),res,label)  
f_tweet.close() 


num_walk=30
max_len=30

training_ips=[]
testing_ips=[]
training=[]
testing=[]
training_rw=[]
testing_rw=[]
y_test=[]
y_train=[]
no_rw=3
test_tids=[]
for i in range(no_rw):
    training.append([])
    testing.append([])
    training_rw.append([])
    testing_rw.append([])

for tid in tweet_net:
    if 'train' in tid and tid in preprocess_corpus_startnode_id and len(preprocess_corpus_startnode_id[tid])>=no_rw:
        samples=[]
        for q1 in tweet_net[tid][1]:
            q1=q1.replace('@','M_').replace('#','H_')
            if q1 in word2idx:  #q1 not in stopwords and 
                samples.append(word2idx[q1])
        training_ips.append(samples)
        for i in range(no_rw):
            training_rw[i].append(convert2id(preprocess_corpus_startnode_id[tid][i][0],word2idx))
        y_train.append(tweet_net[tid][2])
        # for i in range(no_rw):
        #     training[i].append(d2vec[tid][i])     


# for tid in test_tid:
    if 'test' in tid and tid in preprocess_corpus_startnode_id and len(preprocess_corpus_startnode_id[tid])>=no_rw:
        test_tids.append(tid)
        samples=[]
        for q1 in tweet_net[tid][1]:
            q1=q1.replace('@','M_').replace('#','H_')
            if q1 in word2idx:  #q1 not in stopwords and 
                samples.append(word2idx[q1])
        testing_ips.append(samples)
        for i in range(no_rw):
            testing_rw[i].append(convert2id(preprocess_corpus_startnode_id[tid][i][0],word2idx))
        y_test.append(tweet_net[tid][2])



training_ips = sequence.pad_sequences(training_ips, maxlen=max_len)
testing_ips = sequence.pad_sequences(testing_ips, maxlen=max_len)

for i in range(no_rw):
    training_rw[i]=sequence.pad_sequences(training_rw[i], maxlen=max_len)
    testing_rw[i]=sequence.pad_sequences(testing_rw[i], maxlen=max_len)

training1=[training_ips]+training_rw
testing1=[testing_ips]+testing_rw

y_train=convert_labels(y_train)
y_test=convert_labels(y_test)

y_train = np.array(y_train)
y_test = np.array(y_test)



del preprocess_corpus_startnode_id
gc.collect()

emb=len(vectors[0])

input_tweet=len(training1[0][0])
input_node=Input(shape=(input_tweet,))
emb_CNN=CNN(input_node,input_tweet)
# emb_CNN=Dropout(0.2)(emb_CNN)

input_rw_node=[]
emb_rw=[]
input_rw=len(training1[1][0])

for i in range(no_rw):
    input_rw_node.append(Input(shape=(input_rw,)))
    emb_rw.append(CNN(input_rw_node[i],input_rw))

rw_tweets=Concatenate()(emb_rw)
embed_rw=Dense(60, activation='relu')(rw_tweets)
# embed_rw=Dropout(0.2)(embed_rw)



embed_hash_tweets=Concatenate()([emb_CNN,embed_rw])
# embed_hash_tweets=Dropout(0.2)(embed_hash_tweets)
senti_class=Dense(3,activation='softmax')(embed_hash_tweets)
model=Model([input_node]+input_rw_node,senti_class)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(training1, y_train,
          batch_size=512,
          epochs=10,
          validation_data=(testing1, y_test))



model_json = model.to_json()
with open("model/semeval13_biased_rw_ft_emb_no_expansion_"+str(no_rw)+"_CNN_ensemble.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model/semeval13_biased_rw_ft_emb_no_expansion_"+str(no_rw)+"_CNN_ensemble.h5")
print("Saved model to disk")      

result_LSTM=[]
result_LSTM=model.predict(testing1,verbose=0)
fd = open("model/semeval13_biased_rw_ft_emb_no_expansion_"+str(no_rw)+"_CNN_ensemble","w")
for ii,i in enumerate(result_LSTM):
    fd.write(test_tids[ii]+'\t')
    if y_test[ii][0] == 1:
        fd.write('Positive\t')
    elif y_test[ii][1] == 1:
        fd.write('Negative\t')
    else:
        fd.write('Neutral\t')
        
    if i[0] > i[1]:
        if i[0] > i[2]:
            fd.write('Positive\n')
        else:
            fd.write('Neutral\n')
    elif i[1] > i[2]:
        fd.write('Negative\n')
    else:
        fd.write('Neutral\n')
fd.close()


input_tweet=len(training1[0][0])
input_node=Input(shape=(input_tweet,))
emb_CNN=BiLSTM_CNN(input_node,input_tweet)
# emb_CNN=Dropout(0.2)(emb_CNN)

input_rw_node=[]
emb_rw=[]
input_rw=len(training1[1][0])

for i in range(no_rw):
    input_rw_node.append(Input(shape=(input_rw,)))
    emb_rw.append(BiLSTM_CNN(input_rw_node[i],input_rw))

rw_tweets=Concatenate()(emb_rw)
embed_rw=Dense(60, activation='relu')(rw_tweets)
# embed_rw=Dropout(0.2)(embed_rw)

embed_hash_tweets=Concatenate()([emb_CNN,embed_rw])
# embed_hash_tweets=Dropout(0.2)(embed_hash_tweets)
senti_class=Dense(3,activation='softmax')(embed_hash_tweets)
model=Model([input_node]+input_rw_node,senti_class)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(training1, y_train,
          batch_size=512,
          epochs=10,
          validation_data=(testing1, y_test))


model_json = model.to_json()
with open("model/semeval13_biased_rw_ft_emb_no_expansion_"+str(no_rw)+"_BiLSTM_CNN_ensemble.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model/semeval13_biased_rw_ft_emb_no_expansion_"+str(no_rw)+"_BiLSTM_CNN_ensemble.h5")
print("Saved model to disk")      



result_LSTM=[]
result_LSTM=model.predict(testing1,verbose=0)
fd = open("model/semeval13_biased_rw_ft_emb_no_expansion_"+str(no_rw)+"_BiLSTM_CNN_ensemble","w")
for ii,i in enumerate(result_LSTM):
    fd.write(test_tids[ii]+'\t')
    if y_test[ii][0] == 1:
        fd.write('Positive\t')
    elif y_test[ii][1] == 1:
        fd.write('Negative\t')
    else:
        fd.write('Neutral\t')
        
    if i[0] > i[1]:
        if i[0] > i[2]:
            fd.write('Positive\n')
        else:
            fd.write('Neutral\n')
    elif i[1] > i[2]:
        fd.write('Negative\n')
    else:
        fd.write('Neutral\n')
fd.close()



input_tweet=len(training1[0][0])
input_node=Input(shape=(input_tweet,))
emb_CNN=BiLSTM(input_node,input_tweet)
# emb_CNN=Dropout(0.2)(emb_CNN)

input_rw_node=[]
emb_rw=[]
input_rw=len(training1[1][0])

for i in range(no_rw):
    input_rw_node.append(Input(shape=(input_rw,)))
    emb_rw.append(BiLSTM(input_rw_node[i],input_rw))

rw_tweets=Concatenate()(emb_rw)
embed_rw=Dense(60, activation='relu')(rw_tweets)
# embed_rw=Dropout(0.2)(embed_rw)



embed_hash_tweets=Concatenate()([emb_CNN,embed_rw])
# embed_hash_tweets=Dropout(0.2)(embed_hash_tweets)
senti_class=Dense(3,activation='softmax')(embed_hash_tweets)
model=Model([input_node]+input_rw_node,senti_class)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(training1, y_train,
          batch_size=512,
          epochs=10,
          validation_data=(testing1, y_test))



model_json = model.to_json()
with open("model/semeval13_biased_rw_ft_emb_no_expansion_"+str(no_rw)+"_BiLSTM_ensemble.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model/semeval13_biased_rw_ft_emb_no_expansion_"+str(no_rw)+"_BiLSTM_ensemble.h5")
print("Saved model to disk")      



result_LSTM=[]
result_LSTM=model.predict(testing1,verbose=0)
fd = open("model/semeval13_biased_rw_ft_emb_no_expansion_"+str(no_rw)+"_BiLSTM_ensemble","w")
for ii,i in enumerate(result_LSTM):
    fd.write(test_tids[ii]+'\t')
    if y_test[ii][0] == 1:
        fd.write('Positive\t')
    elif y_test[ii][1] == 1:
        fd.write('Negative\t')
    else:
        fd.write('Neutral\t')
        
    if i[0] > i[1]:
        if i[0] > i[2]:
            fd.write('Positive\n')
        else:
            fd.write('Neutral\n')
    elif i[1] > i[2]:
        fd.write('Negative\n')
    else:
        fd.write('Neutral\n')
fd.close()



input_tweet=len(training1[0][0])
input_node=Input(shape=(input_tweet,))
emb_CNN=BiLSTM_self_attention_CNN(input_node,input_tweet)
# emb_CNN=Dropout(0.2)(emb_CNN)

input_rw_node=[]
emb_rw=[]
input_rw=len(training1[1][0])

for i in range(no_rw):
    input_rw_node.append(Input(shape=(input_rw,)))
    emb_rw.append(BiLSTM_self_attention_CNN(input_rw_node[i],input_rw))

rw_tweets=Concatenate()(emb_rw)
embed_rw=Dense(60, activation='relu')(rw_tweets)
# embed_rw=Dropout(0.2)(embed_rw)



embed_hash_tweets=Concatenate()([emb_CNN,embed_rw])
# embed_hash_tweets=Dropout(0.2)(embed_hash_tweets)
senti_class=Dense(3,activation='softmax')(embed_hash_tweets)
model=Model([input_node]+input_rw_node,senti_class)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(training1, y_train,
          batch_size=512,
          epochs=10,
          validation_data=(testing1, y_test))


model_json = model.to_json()
with open("model/semeval13_biased_rw_ft_emb_no_expansion_"+str(no_rw)+"_BiLSTM_self_attention_CNN_ensemble.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model/semeval13_biased_rw_ft_emb_no_expansion_"+str(no_rw)+"_BiLSTM_self_attention_CNN_ensemble.h5")
print("Saved model to disk")      



result_LSTM=[]
result_LSTM=model.predict(testing1,verbose=0)
fd = open("model/semeval13_biased_rw_ft_emb_no_expansion_"+str(no_rw)+"_BiLSTM_self_attention_CNN_ensemble","w")
for ii,i in enumerate(result_LSTM):
    fd.write(test_tids[ii]+'\t')
    if y_test[ii][0] == 1:
        fd.write('Positive\t')
    elif y_test[ii][1] == 1:
        fd.write('Negative\t')
    else:
        fd.write('Neutral\t')
        
    if i[0] > i[1]:
        if i[0] > i[2]:
            fd.write('Positive\n')
        else:
            fd.write('Neutral\n')
    elif i[1] > i[2]:
        fd.write('Negative\n')
    else:
        fd.write('Neutral\n')
fd.close()
