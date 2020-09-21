import networkx as nx
import pandas as pd
H = nx.read_gpickle("data/corpus_graph_influence3.gpickle")

import numpy as np
import gc
import sys

import pickle
from gensim.models.fasttext import FastText

stopwords = ['rt','amp','url','sir','day','title','shri','crore','time',"a", "about","above", "across", "after", "afterwards", "again", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]
model = FastText.load('model/semeval13_multiplex_walk_fastext.model')


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

keywords=[]
centrality_score={}
for n in H.nodes(data=True):
    d = dict()
    d['node_influence'] = n[1]['node_influence']
    d['layer_id'] = n[1]['layer_id']
    d['layer_influence'] = H.graph['layer_influence'][H.graph['layer_to_id'].index(n[1]['layer_id'])]
    centrality_score[n[1]['node_id']]=d
    if n[1]['node_id'] not in stopwords and len(n[1]['node_id'])>1:
        keywords.append(n[1]['node_id'])

word_similarity={}
for word in keywords:
    word_similarity[word]=model.wv.most_similar([word], topn=5)
    print(word,word_similarity[word])



import math
import editdistance
import pandas as pd


def edit_distance_pc(ipword1,ipword2):
    w1=ipword1.replace('M_','').replace('H_','')
    w2=ipword2.replace('M_','').replace('H_','')   
    xx=editdistance.eval(w1,w2)
    #print(w1,w2)
    mm=min(len(w1),len(w2))
    mx=max(len(w1),len(w2))
    cor=mx-xx
    if cor>0:
        xx=math.log(cor/mx)+math.log(cor/mm)
    else:
        xx=math.log(0.000001/mx)+math.log(0.000001/mm)
    return xx
    
def logistic(total):
    total=1/(1+math.exp(-total))
    return total

edit_score={}
for word in word_similarity:
    w1=word.replace('M_','').replace('H_','')
    if w1 !='':
        edit_score[word]={}
        edit_score[word]['centrality_score']=centrality_score[word]
        edit_score[word]['Top_Similarity']={}
        for sim_w in word_similarity[word]:
            xx=0
            w2=sim_w[0].replace('M_','').replace('H_','')
            if sim_w[0] not in stopwords and w2!='':
                if sim_w not in edit_score[word]['Top_Similarity']:
                    edit_score[word]['Top_Similarity'][sim_w[0]]={}
                xx=edit_distance_pc(w1,w2)
                edit_score[word]['Top_Similarity'][sim_w[0]]['edit_distance']=xx
                if sim_w[0] in centrality_score:
                    edit_score[word]['Top_Similarity'][sim_w[0]]['centrality_score']=centrality_score[sim_w[0]]
                else:
                    edit_score[word]['Top_Similarity'][sim_w[0]]['centrality_score']=None
                edit_score[word]['Top_Similarity'][sim_w[0]]['cosine_score']=sim_w[1]

del word_similarity

node_information={}
for word in edit_score:
    df=pd.DataFrame(columns=['similar_word','edit_score','cosine_score','node_influence','layer_influence'])
    for ii,i in enumerate(edit_score[word]['Top_Similarity']):
        if edit_score[word]['Top_Similarity'][i]['centrality_score'] is not None:
            df.loc[ii]=[i,logistic(edit_score[word]['Top_Similarity'][i]['edit_distance']),logistic(edit_score[word]['Top_Similarity'][i]['cosine_score']),logistic(edit_score[word]['Top_Similarity'][i]['centrality_score']['node_influence']),logistic(edit_score[word]['Top_Similarity'][i]['centrality_score']['layer_influence'])]
        else:
            df.loc[ii]=[i,logistic(edit_score[word]['Top_Similarity'][i]['edit_distance']),logistic(edit_score[word]['Top_Similarity'][i]['cosine_score']),logistic(float(0)),logistic(float(0))]
    node_information[word]=df

import pickle as pkl 
f = open('data/centraliti_merged.pkl', 'wb') 
pkl.dump(node_information, f)
f.close()