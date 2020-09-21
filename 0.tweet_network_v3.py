# -*- coding: utf-8 -*-
import re
import numpy as np
import nltk
from nltk.tokenize import TweetTokenizer
import sys
import string
import re

tstop = ['rt','amp','url']

import itertools   

ecount=0
ccount=0

def text_cleaner(text):
    newString = text.lower() 
    newString = re.sub(r"'s\b"," ",newString)
    newString = re.sub("[^a-zA-Z_@#]", " ", newString) 
    long_words=[]
    for i in newString.split():
        if i not in tstop:                  
            long_words.append(i)
    return long_words


def preprocess_tweet_network(tweet):
    edges=[]
    prev=''
    prev_meta=''
    prev_type=0 #keyword:0 hashtag:1, mention:2
    hashtags=[]
    mentions=[]
    h_k=[]
    m_k=[]
    ii=0
    res=text_cleaner(tweet)
    res_update=[]
    regex = r'https?:|urls?|[/\:,-."\'?!;…]+'
    print(tid,res)
    for lenp,phrases in enumerate(res):
        meta=0  
        if '@' in phrases:
            phrases='M_'+phrases.replace('@','')
            mentions.append(phrases)
            prev_meta=phrases
            meta=2
        elif '#' in phrases:
            phrases='H_'+phrases.replace('#','')
            hashtags.append(phrases)
            prev_meta=phrases
            meta=1
        
        if meta>0 and lenp>0:
            if meta==2 and prev !='':
                m_k.append((prev,phrases))
            elif prev !='':
                h_k.append((prev,phrases))
        
        elif meta==0:
            pres=text_cleaner(phrases)
            if len(pres) >0:
                for wrd in pres:
                    wrd = re.sub(regex,'', wrd)
                    if wrd != '' and len(wrd)>1:
                        if ii==0:
                            prev=wrd
                            ii+=1
                        else:
                            if prev!='' and len(prev)>1:
                                edge=(prev,wrd)
                                edges.append(edge)
                                prev=wrd  
                            else:
                                prev=wrd
                        if prev_type == 2:
                            m_k.append((prev_meta,wrd))
                        elif prev_type == 1:
                            h_k.append((prev_meta,wrd))
                    else:
                        edges.append(prev)
                        prev=''
            else:
                if prev != '' and len(prev)>1:
                    edges.append(prev)
                    prev=''
                
        else:
            print("Issue here",phrases)
        prev_type=meta
    print('Tweet: ',tweet,'\nKeyword Egdes:',edges,'\nHashtag Keyword Egdes:',h_k,'\nMention Keyword Egdes:',m_k)
    return ((hashtags,mentions,edges,h_k,m_k),res,label)

import emoji

def remove_link(text):
    regex = r'https?://[^\s<>)"‘’]+'
    match = re.sub(regex,' ', text)
    regex = r'urls?'
    match = re.sub(regex,' ', match)
    tweet=emoji.demojize(match) 
    tweet = re.sub("[^a-zA-Z_@#]", " ", tweet)
    tweet = re.sub("[ ]+", " ", tweet) 
    return tweet


f = open("data/tweet_matrix_semeval13_train","r")
sentences=f.readlines()
f.close()

tweet_net=dict()
for tid,line in enumerate(sentences):
    line=line.strip().split('\t')
    label=line[1]
    tweet=line[0].lower().replace('@',' @').replace('#',' #')
    tid="train_"+str(tid)
    tweet=remove_link(tweet)
    res=tweet.split()
    (hashtags,mentions,edges,h_k,m_k),res,label=preprocess_tweet_network(tweet)
    tweet_net[tid]=((hashtags,mentions,edges,h_k,m_k),res,label)

f = open("data/tweet_matrix_semeval13_test","r")
sentences=f.readlines()
f.close()

for tid,line in enumerate(sentences):
    line=line.strip().split('\t')
    label=line[1]
    tweet=line[0].lower().replace('@',' @').replace('#',' #')
    tid="test_"+str(tid)
    tweet=remove_link(tweet)
    res=tweet.split()
    (hashtags,mentions,edges,h_k,m_k),res,label=preprocess_tweet_network(tweet)
    tweet_net[tid]=((hashtags,mentions,edges,h_k,m_k),res,label)


len(tweet_net)

import pickle as pkl 
f = open('data/semeval13_multiplex_network.pkl', 'wb') 
pkl.dump(tweet_net, f)
f.close()