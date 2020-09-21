# -*- coding: utf-8 -*-
import re
import numpy as np
import nltk
from nltk.tokenize import TweetTokenizer
#from nltk.corpus import stopwords
#from nltk.tokenize import RegexpTokenizer
import sys
#from pattern.en import parse
import string

stopwords = ['rt','amp','url','sir','day','title','shri','crore','time',"a", "about","above", "across", "after", "afterwards", "again", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]
tstop = ['rt','amp','url']

import itertools   

ecount=0
ccount=0

import re

def text_cleaner(text):
    newString = text
    newString = re.sub(r"'s\b"," ",newString)
    newString = re.sub("[^a-zA-Z_@#]", " ", newString) 
    long_words=[]
    for i in newString.split():
        if i not in tstop:                  
            long_words.append(i)
    return long_words

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

def normalized_lm(text):
	wlist=[]
	for w in text:
		if 'H_' in w:
			wlist.append('Hashtag')
		elif 'M_' in w:
			wlist.append('Mentioned')
		else:
			w=w.replace('_',' ').split()
			wlist+=w
	return wlist
  
import pickle
f_tweet = open('data/semeval13_multiplex_network.pkl', 'rb')
tweet_net=pickle.load(f_tweet)       #((hashtags,mentions,edges,h_k,m_k),res,label)
f_tweet.close()


tweet_corpus=[]
for tid in tweet_net:
    line=' '.join(tweet_net[tid][1])  #line.strip().split('\t')
    label=tweet_net[tid][2]
    #tid="T_"+str(tid)
    tweet=line.lower().replace('@',' @').replace('#',' #')
    tweet=remove_link(tweet)
    tweet=text_cleaner(tweet.replace('@',' M_').replace('#',' H_'))
    tweet=normalized_lm(tweet)
    tweet_corpus.append(tweet)


from nltk import bigrams, trigrams
from collections import Counter, defaultdict

# Create a placeholder for model
model = defaultdict(lambda: defaultdict(lambda: 0))

# Count frequency of co-occurance  
for sentence in tweet_corpus:
    for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
        model[(w1, w2)][w3] += 1
 
# Let's transform the counts to probabilities
for w1_w2 in model:
    total_count = float(sum(model[w1_w2].values()))
    for w3 in model[w1_w2]:
        model[w1_w2][w3] /= total_count



for value in model:
    model[value] = dict(model[value])
model = dict(model)

import pickle as pkl 
f = open('model/semeval13_statistical_model.pkl', 'wb') 
pkl.dump(model, f)
f.close()


