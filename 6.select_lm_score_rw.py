import networkx as nx
import numpy as np
import gc
import sys
import pickle

stopwords = ['rt','amp','url','https','sir','day','title','shri','crore','time',"a", "about","above", "across", "after", "afterwards", "again", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]
tstop = ['rt','amp','url']

def get_weight_matrix(words,mx_model):
	# define weight matrix dimensions with all 0
	idx=1
	word2idx={}
	vectors=[]
	vectors.append(np.asarray([0]*mx_model.vector_size))
	for i in words:
		word2idx[i]=idx#print(i,len(w2v[i]))
		vectors.append(np.asarray(mx_model[i]))
		idx+=1
	return word2idx,vectors

import pickle as pkl


import re
def text_cleaner(text):
	newString = text.lower() 
	newString=newString.replace('@','M_').replace('#','H_')
	newString = re.sub(r"'s\b"," ",newString)
	newString = re.sub("[^a-zA-Z_@#]", " ", newString) 
	long_words=[]
	for i in newString.split():
		if i not in tstop:				  
			long_words.append(i)
	return long_words


import multiprocessing 


import pandas as pd
def select_K_SHE_lm(xx):
	(rw_walks,f)=xx
	preprocess_corpus_startnode_id={}
	df=pd.DataFrame(columns=['tweet','LM_score'])
	cur_tid=''
	dfi=0
	start_nodes={}
	for walk in rw_walks:
		tid,tweet,c_score=walk.strip().split('\t')
		tweet=tweet.split()
		if cur_tid=='':
			cur_tid=tid
			df.loc[dfi]=[tweet,c_score]
			dfi+=1
		elif cur_tid!=tid:
			print(tid,tweet,c_score)
			if cur_tid not in preprocess_corpus_startnode_id:
				preprocess_corpus_startnode_id[cur_tid]=[]

			df.sort_values(by=['LM_score'], inplace=True, ascending=False)
			for i in df.index[:6]:
				df_ids=df['tweet'][i]
				preprocess_corpus_startnode_id[cur_tid].append((df_ids,c_score))
			cur_tid=tid
			dfi=0
			df=pd.DataFrame(columns=['tweet','LM_score'])
			df.loc[dfi]=[tweet,c_score]
			dfi+=1

		else:
			df.loc[dfi]=[tweet,c_score]
			dfi+=1
	
	pkl.dump(preprocess_corpus_startnode_id, f)
	f.close()

f1 = open('data/semeval13_biased_rw_startnode_without_expansion_lm.pkl', 'wb') 
f=open('data/semeval13_biased_rw_startnode_without_expansion_lm','r')
rw_walks1=f.readlines()
f.close()
select_K_SHE_lm((rw_walks1,f1))
