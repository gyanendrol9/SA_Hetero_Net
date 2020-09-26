import pickle

f_tweet = open('data/semeval13_multiplex_network.pkl', 'rb')
tweet_net=pickle.load(f_tweet)       #((hashtags,mentions,edges,h_k,m_k),res,label)
f_tweet.close()

from collections import defaultdict
# Create a placeholder for model
kkmodel = defaultdict(lambda: defaultdict(lambda: 0))
hkmodel = defaultdict(lambda: defaultdict(lambda: 0))
mkmodel = defaultdict(lambda: defaultdict(lambda: 0))
khmodel = defaultdict(lambda: defaultdict(lambda: 0))
kmmodel = defaultdict(lambda: defaultdict(lambda: 0))
hmmodel = defaultdict(lambda: defaultdict(lambda: 0))
hhmodel = defaultdict(lambda: defaultdict(lambda: 0))
mmmodel = defaultdict(lambda: defaultdict(lambda: 0))

def addup(edge_dict,new_edges):
    for edge in new_edges:
        if edge not in edge_dict:
            edge_dict[edge]=0
        edge_dict[edge]+=1

import itertools

for tid in tweet_net:
    hashtags=tweet_net[tid][0][0]
    mentions=tweet_net[tid][0][1]
    H_M = list(itertools.product(set(hashtags), set(mentions)))
    H_H=[(x) for x in itertools.permutations(set(hashtags),2)]
    M_M=[(x) for x in itertools.permutations(set(mentions),2)]
    K_K=tweet_net[tid][0][2]
    H_K=[]
    K_H=[]
    for i in tweet_net[tid][0][3]:
        node0=i[0]
        node1=i[1]
        if 'H_' in i[0]:
            H_K.append((node0,node1))
        else:
            K_H.append((node0,node1))  
    M_K=[]
    K_M=[]
    for i in tweet_net[tid][0][4]:
        node0=i[0]
        node1=i[1]
        if 'M_' in i[0]:
            M_K.append((node0,node1))
        else:
            K_M.append((node0,node1))  
    addup(hhmodel,H_H)
    addup(hkmodel,H_K)
    addup(hmmodel,H_M)
    addup(kkmodel,K_K)
    addup(mkmodel,M_K)
    addup(kmmodel,K_M)
    addup(khmodel,K_H)
    addup(mmmodel,M_M)



def create_edges(f,model):
    for edge in model:
        if len(edge)==2:
            if len(edge[0])>1:
                f.write(edge[0]+","+edge[1]+','+str(model[edge])+"\n")


f=open('data/cooccurance_hashtags.csv','w')
create_edges(f,hhmodel)
f.close()
f=open('data/cooccurance_mentions.csv','w')
create_edges(f,mmmodel)
f.close()
f=open('data/cooccurance_keywords.csv','w')
create_edges(f,kkmodel)
f.close()
f=open('data/hashtags_keywords.csv','w')
create_edges(f,hkmodel)
f.close()
f=open('data/mentions_keywords.csv','w')
create_edges(f,mkmodel)
f.close()
f=open('data/hashtags_mentions.csv','w')
create_edges(f,hmmodel)
f.close()
f=open('data/keywords_hashtags.csv','w')
create_edges(f,khmodel)
f.close()
f=open('data/keywords_mentions.csv','w')
create_edges(f,kmmodel)
f.close()