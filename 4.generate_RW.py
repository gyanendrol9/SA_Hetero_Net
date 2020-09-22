import pickle
import itertools
import networkx as nx
import numpy as np
import pandas as pd
import time
import random
from collections import OrderedDict
from decimal import Decimal
from multiplexRW import * 
import numpy as np
import gc
import sys
import Random_walk
import re
import math
import editdistance
from gensim.models.fasttext import FastText

stopwords = ['rt','amp','url','https','sir','day','title','shri','crore','time',"a", "about","above", "across", "after", "afterwards", "again", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]


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

def similar_word_merge(ipword,s_type='edit_score'):
    selected=[]
    node_information[ipword].sort_values(by=['edit_score'], inplace=True,ascending=False)
    for i in node_information[ipword]['similar_word'][:2]:
        selected.append(i)

    node_information[ipword].sort_values(by=['cosine_score'], inplace=True,ascending=False)
    for i in node_information[ipword]['similar_word'][:2]:
        if i not in selected:
            selected.append(i)

    node_information[ipword].sort_values(by=['node_influence'], inplace=True,ascending=False)
    for i in node_information[ipword]['similar_word'][:2]:
        if i not in selected:
            selected.append(i)

    node_information[ipword].sort_values(by=['layer_influence'], inplace=True,ascending=False)
    for i in node_information[ipword]['similar_word'][:2]:
        if i not in selected:
            selected.append(i)

    final_selected=selected


    new_selected=[]
    for word in selected:
        try:
            node_information[word].sort_values(by=['edit_score'], inplace=True,ascending=False)
            for i in node_information[word]['similar_word'][:2]:
                if i not in final_selected:
                    new_selected.append(i)

            node_information[word].sort_values(by=['cosine_score'], inplace=True,ascending=False)
            for i in node_information[word]['similar_word'][:2]:
                if i not in final_selected and i not in new_selected:
                    new_selected.append(i)

            node_information[word].sort_values(by=['node_influence'], inplace=True,ascending=False)
            for i in node_information[word]['similar_word'][:2]:
                if i not in final_selected and i not in new_selected:
                    new_selected.append(i)

            node_information[word].sort_values(by=['layer_influence'], inplace=True,ascending=False)
            for i in node_information[word]['similar_word'][:2]:
                if i not in final_selected and i not in new_selected:
                    new_selected.append(i)
        except:
            pass

    final_selected+=list(set(new_selected))

    df=pd.DataFrame(columns=['similar_word','edit_score','cosine_score','node_influence','layer_influence','total'])
    for ii,selected in enumerate(final_selected):
        w1=ipword.replace('M_','').replace('H_','')
        w2=selected.replace('M_','').replace('H_','')
        xx=edit_distance_pc(w1,w2)
        cosined=model.wv.similarity(ipword, selected)
        if selected in centrality_score:
            node_influence=centrality_score[selected]['node_influence']
            layer_influence=centrality_score[selected]['layer_influence']
        else:
            node_influence=float(0)
            layer_influence=float(0)
        total=0.5*logistic(node_influence)+0.5*logistic(layer_influence)
        df.loc[ii]=[selected,xx,cosined,logistic(node_influence),logistic(layer_influence),total]

    if s_type=='edit_score':
        df.sort_values(by=[s_type], inplace=True, ascending=False)
    else:
        df.sort_values(by=[s_type], inplace=True, ascending=False)

    return df


def get_similar_nodes(query,final_information,score):
    final_selected=[]
    if not final_information.empty:
        final_information.sort_values(by=['edit_score'], inplace=True, ascending=False)
        for i in final_information.index:
            if final_information['similar_word'][i] not in final_selected and final_information['similar_word'][i]!=query:
                if final_information['edit_score'][i]>score:
                    final_selected.append(final_information['similar_word'][i])
                    #print(final_information['similar_word'][i],final_information['edit_score'][i])
            

        final_information.sort_values(by=['total'], inplace=True, ascending=False)
        for i in final_information['similar_word'][:2]:
            if i not in final_selected and i!=query:
                final_selected.append(i)

    return final_selected

def retrieve_query(query,final_information,sortby='edit_score'):
    df=pd.DataFrame(columns=['similar_word','edit_score','centrality_score','total'])
    for ii,i in enumerate(final_information):
        xx=edit_distance_pc(query,i)
        cosined=model.wv.similarity(query, i)
        if i in centrality_score:
            total=0.5*centrality_score[i]['node_influence']+0.5*centrality_score[i]['layer_influence']
            df.loc[ii]=[i,xx*cosined,centrality_score[i]['node_influence'],total]
        else:
            total=0.0001
            total=1/(1+math.exp(-total))
            df.loc[ii]=[i,xx*cosined,-0.0001,total]
        
    df.sort_values(by=[sortby], inplace=True, ascending=False)
    return df


def merger(query):
    result=[query]
    if query in node_information:
        x=similar_word_merge(query)
        result=get_similar_nodes(query,x,-1)
        final_set=[]
        prev_set=[]
        old=0
        new=len(result)
        loop=0
        while old<new:
            for node in result:
                if node not in prev_set:
                    if node in node_information:
                        x=similar_word_merge(node)
                        final_set+=get_similar_nodes(node,x,-1)
                        final_set=list(set(final_set))
            prev_set=result
            old=len(result)
            result=final_set
            new=len(result)
            loop+=1
    return result



def get_top(pdresult,k=1,sortby='edit_score'):
    top=[]
    pdresult.sort_values(by=[sortby], inplace=True, ascending=False)
    if not pdresult.empty:
        for i in pdresult['similar_word']:
            if len(top)<k and len(i)>1:
                top.append(i)
            elif len(top)>k:
                return top
    return top


def get_k_similarity(queries,k,sortby='edit_score'):
    x=list(queries)
    xx=[]
    for query in queries:
        print(query)
        if query in node_information:
            for i in node_information[query]['similar_word']:
                x.append(i)
                cosined=model.wv.similarity(query,i)
                xx.append((i,cosined))
    xx+=model.wv.most_similar(x, topn=5)
    result=[]
    for i in xx:
        result.append(i[0])
    pdresult=retrieve_query(query,set(result))   
    result=get_top(pdresult,k,sortby)
    result=list(set(result))
    return (result,pdresult)



#reload(sys)
#sys.setdefaultencoding('utf-8')

#python code.py 5 #number of walk


def display_time_taken(time_taken):
    time_str = ''
    if time_taken > 3600:
        hr = time_taken / 3600
        time_taken = time_taken % 3600
        time_str += str(int(hr)) + 'h '
    if time_taken > 60:
        mi = time_taken / 60
        time_taken = time_taken % 60
        time_str += str(int(mi)) + 'm '
    time_str += str(int(time_taken)) + 's'
    print('Time Taken: %s' % time_str)


def get_alias_edge(src, dst):
    unnormalized_probs = [];  # p=0.3; q=0.7
    for dst_nbr in sorted(H.neighbors(dst)):
        if dst_nbr == src:
            unnormalized_probs.append(
                H[dst][dst_nbr][0]['weight'] / p if 0 in H[dst][dst_nbr] else H[dst][dst_nbr]['weight'] / p)
        elif H.has_edge(dst_nbr, src):
            unnormalized_probs.append(
                H[dst][dst_nbr][0]['weight'] if 0 in H[dst][dst_nbr] else H[dst][dst_nbr]['weight'])
        else:
            unnormalized_probs.append(
                H[dst][dst_nbr][0]['weight'] / q if 0 in H[dst][dst_nbr] else H[dst][dst_nbr]['weight'] / q)
    norm_const = sum(unnormalized_probs)
    normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

    return alias_setup(normalized_probs)


def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def preprocess_tweet_transition_probs(X, Z, node_num_dict, layer_dict):
    global alias_nodes, alias_edges
    alias_nodes = {}

    if X is not None and Z is not None:
        sort_keys = layer_dict
        if (len(node_num_dict)) == 2:
            k1 = sort_keys[0];
            n1 = node_num_dict[0]
            k2 = sort_keys[1];
            n2 = node_num_dict[1]
            X[0:n1] = np.multiply(Z[0], X[0:n1])
            X[n1:n1 + n2] = np.multiply(Z[1], X[n1:n1 + n2])
        elif (len(sort_keys)) == 3:
            k1 = sort_keys[0];
            n1 = node_num_dict[0]
            k2 = sort_keys[1];
            n2 = node_num_dict[1]
            k3 = sort_keys[2];
            n3 = node_num_dict[2]
            X[0:n1] = np.multiply(Z[0], X[0:n1])
            X[n1:n1 + n2] = np.multiply(Z[1], X[n1:n1 + n2])
            X[n1 + n2:] = np.multiply(Z[2], X[n1 + n2:])
        X = np.divide(np.array(X), np.array(X).sum())

    for node in H.nodes():
        unnormalized_probs = [r * H[node][nbr][0]['weight'] if 0 in H[node][nbr] else H[node][nbr]['weight'] for nbr in
                              sorted(H.neighbors(node))]
        if X is not None and Z is not None:
            restart_probs = [(1 - r) * X[nbr] for nbr in sorted(H.neighbors(node))]
            final_probs = sum(restart_probs, unnormalized_probs)
        else:
            final_probs = unnormalized_probs
        norm_const = sum(final_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in final_probs]
        alias_nodes[node] = alias_setup(normalized_probs)

    alias_edges = {}
    triads = {}
    if nx.is_directed(H):
        for edge in H.edges():
            alias_edges[edge] = get_alias_edge(edge[0], edge[1])
    else:
        for edge in H.edges():
            alias_edges[edge] = get_alias_edge(edge[0], edge[1])
            alias_edges[(edge[1], edge[0])] = get_alias_edge(edge[1], edge[0])

    return


def preprocess_transition_probs(X=None, Z=None):
    global alias_nodes, alias_edges
    alias_nodes = {}

    if X is not None and Z is not None:
        sort_keys = sorted(h.multiplexes.keys())
        if (len(sort_keys)) == 2:
            k1 = sort_keys[0];
            n1 = h.node_num_dict[k1]
            k2 = sort_keys[1];
            n2 = h.node_num_dict[k2]
            X[0:n1] = np.multiply(Z[0], X[0:n1])
            X[n1:n1 + n2] = np.multiply(Z[1], X[n1:n1 + n2])
        elif (len(sort_keys)) == 3:
            k1 = sort_keys[0];
            n1 = h.node_num_dict[k1]
            k2 = sort_keys[1];
            n2 = h.node_num_dict[k2]
            k3 = sort_keys[2];
            n3 = h.node_num_dict[k3]
            X[0:n1] = np.multiply(Z[0], X[0:n1])
            X[n1:n1 + n2] = np.multiply(Z[1], X[n1:n1 + n2])
            X[n1 + n2:] = np.multiply(Z[2], X[n1 + n2:])
        X = np.divide(np.array(X), np.array(X).sum())

    for node in H.nodes():
        unnormalized_probs = [r * H[node][nbr][0]['weight'] if 0 in H[node][nbr] else H[node][nbr]['weight'] for nbr in
                              sorted(H.neighbors(node))]
        if X is not None and Z is not None:
            restart_probs = [(1 - r) * X[nbr] for nbr in sorted(H.neighbors(node))]
            final_probs = sum(restart_probs, unnormalized_probs)
        else:
            final_probs = unnormalized_probs
        norm_const = sum(final_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in final_probs]
        alias_nodes[node] = alias_setup(normalized_probs)

    alias_edges = {}
    triads = {}
    if nx.is_directed(H):
        for edge in H.edges():
            alias_edges[edge] = get_alias_edge(edge[0], edge[1])
    else:
        for edge in H.edges():
            alias_edges[edge] = get_alias_edge(edge[0], edge[1])
            alias_edges[(edge[1], edge[0])] = get_alias_edge(edge[1], edge[0])

    return


def walks(start_node, tweet_nodeList=None):
    walk_list = list()
    for walk_iter in range(num_walks):
        source = start_node
        walk = list();
        walk.append(source)
        for w_length in range(walk_length):
            cur = walk[-1]
            cur_nbrs = sorted(H.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2];
                    flag = True;
                    count = 0
                    while flag == True:
                        next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]
                        count = count + 1
                        if prev != next:
                            flag = False
                        elif prev == next and count == 4:
                            break
                    if count == 4 and flag == True:
                        next = random.choice(list(H.neighbors(cur)))
                    # print("prev, current, next ",prev, cur, next)
                    walk.append(next)
            else:
                break
        if tweet_nodeList is None:
            l = ' '.join(str(h.pool_of_nodes[i]) for i in walk)
        else:
            l = ' '.join(str(tweet_nodeList[i]) for i in walk)
        walk_list.append(l)
    walk_str = '\n'.join(str(w) for w in walk_list)
    walk_str += "\n"

    #for line in walk_str:
        #f.write(line)

    return walk_str, walk_list


def simulate_walks(layer_files, bipartite_files):
    multiplex_dict = dict.fromkeys(layer_files)
    for key, value in layer_files.items():
        multiplex_dict[key] = multilayer({key: value})
        multiplex_dict[key].dump_info()

    global h, H
    h = hetero_multilayer(multiplex_dict, bipartite_files)
    H = nx.from_scipy_sparse_matrix(h.supra_transition_matrix, edge_attribute='weight', create_using=nx.MultiDiGraph())

    X, Z = MultiRank(h)

    walk_df = []
    for k, _ in Z.items():
        preprocess_transition_probs(X[k], Z[k])

        node_list = H.nodes()
        print("Printing the Walks!\n")
        for i in node_list:
            walk_str, walk_list = walks(i)
            print(walk_str);
            d = OrderedDict()
            d['gamma'] = round(k * 0.1, 3)
            d['layer_centrality'] = ['{:.2e}'.format(Decimal(itm)) for itm in Z[k]]
            d['node_centrality'] = ['{:.2e}'.format(Decimal(itm)) for itm in X[k]]
            d['p'] = p
            d['q'] = q
            d['num_walks'] = num_walks,
            d['walk_length'] = walk_length
            d['restart'] = round(r, 2)
            d['jump'] = delta
            d['walk'] = walk_list
            walk_df.append(d)

    walk_df = pd.DataFrame(walk_df)
    #walk_df.to_csv("walks.csv", index=False)

    return walk_df


def simulate_normal_walk(layer_files):
    multiplex_dict = dict.fromkeys(layer_files)
    for key, value in layer_files.items():
        multiplex_dict[key] = multilayer({key: value})
        multiplex_dict[key].dump_info()

    global h, H
    h = hetero_multilayer(multiplex_dict, None)
    H = nx.from_scipy_sparse_matrix(h.supra_transition_matrix, edge_attribute='weight', create_using=nx.MultiDiGraph())

    preprocess_transition_probs()

    walk_df = []
    node_list = H.nodes()
    print("Printing the Walks!\n")
    for i in node_list:
        walk_str, walk_list = walks(i)
        print(walk_str);
        print()

        d = OrderedDict()
        d['p'] = p
        d['q'] = q
        d['num_walks'] = num_walks,
        d['walk_length'] = walk_length
        d['restart'] = round(r, 2)
        d['jump'] = delta
        d['walk'] = walk_list
        walk_df.append(d)

    walk_df = pd.DataFrame(walk_df)
    #walk_df.to_csv("walks.csv", index=False)

    return walk_df


def simulate_tweet_rw_from_original_graph(tweet_augmented):
    walk_df = []
    global H

    graph_df = []
    for n in tweet_augmented.nodes(data=True):
        d = dict()
        d['orig_node_id'] = n[0]
        d['node_id'] = n[1]['node_id']
        d['node_influence'] = n[1]['node_influence']
        d['layer_id'] = n[1]['layer_id']
        d['layer_influence'] = tweet_augmented.graph['layer_influence'][
            tweet_augmented.graph['layer_to_id'].index(n[1]['layer_id'])]
        graph_df.append(d)
    graph_df = pd.DataFrame(graph_df)
    graph_df = graph_df.sort_values(by='layer_id')
    graph_df.reset_index(level=0, inplace=True)

    A = nx.to_scipy_sparse_matrix(tweet_augmented, nodelist=list(graph_df['orig_node_id']))

    node_num_dict = list(graph_df.groupby(['layer_id']).count()['node_id'])
    layer_dict = list(graph_df.groupby(['layer_id']).count()['node_id'].index.values)
    X = list(graph_df['node_influence'])
    Z = tweet_augmented.graph['layer_influence']

    H = nx.from_scipy_sparse_matrix(A, create_using=nx.DiGraph())
    preprocess_tweet_transition_probs(X, Z, node_num_dict, layer_dict)

    node_list = H.nodes()
    #print("Printing the Walks!\n")
    for i in node_list:
        walk_str, walk_list = walks(i, tweet_nodeList=list(graph_df['node_id']))
        #print(walk_str);
        d = OrderedDict()
        d['gamma'] = round(choice_of_gamma)
        d['layer_centrality'] = ['{:.2e}'.format(Decimal(itm)) for itm in Z]
        d['node_centrality'] = ['{:.2e}'.format(Decimal(itm)) for itm in X]
        d['p'] = p
        d['q'] = q
        d['num_walks'] = num_walks,
        d['walk_length'] = walk_length
        d['restart'] = round(r, 2)
        d['jump'] = delta
        d['walk'] = walk_list
        walk_df.append(d)

    walk_df = pd.DataFrame(walk_df)
    #walk_df.to_csv("walks.csv", index=False)

    return walk_df


f_tweet = open('data/semeval13_multiplex_network.pkl', 'rb')
tweet_net=pickle.load(f_tweet)       #((hashtags,mentions,edges,h_k,m_k),res,label)
f_tweet.close()


regex = r'https?:|urls?|[/\:,-."\'?!;’|…]+'

preprocess_corpus_startnode=open('data/semeval13_biased_rw_startnode_without_expansion','w')

tids=0
for tid in tweet_net:
    # tid="T_"+str(tids)
    if len(tweet_net[tid][1])>0:
        print(tid,tweet_net[tid][1],tweet_net[tid][2])
        K_K=[]
        K_missed=[]
        for i in tweet_net[tid][0][2]:
            if i != '' and len(i)==2:
                if len(i[0])>1 and i[0] != i[1]:
                    K_K.append(i)
                elif i[0] != i[1]:
                    print(i)
                    K_missed.append(i)
            else:
                print(i)
                K_missed.append(i)

        H_K=[]
        keywordList=[]
        for i in tweet_net[tid][0][3]:
            node0=''
            node1=''
            if 'H_' in i[0]:
                node0=i[0]
                node1=i[1]
            else:
                node0=i[1]
                node1=i[0]
            keywordList.append(node1)
            H_K.append((node0,node1))

        M_K=[]
        for i in tweet_net[tid][0][4]:
            node0=''
            node1=''
            if 'M_' in i[0]:
                node0=i[0]
                node1=i[1]
            else:
                node0=i[1]
                node1=i[0]
            keywordList.append(node1)
            M_K.append((node0,node1))

        for i in K_K:
            if i != '':
                if len(i[0])>1:
                    keywordList.append(i[0])
                if len(i[1])>1:
                    keywordList.append(i[1])
        keywordList=list(set(keywordList))
        hashtagList=tweet_net[tid][0][0]
        mentionList=tweet_net[tid][0][1]

        query_keyword=[]
        for q1 in keywordList:
            if q1 not in stopwords:
                query_keyword.append(q1)

        #node expansion module can be added here

        if len(K_K)==0 or len(keywordList)<2:
            K_K+=[(x) for x in itertools.permutations(set(keywordList),2)]

        H_M = list(itertools.product(set(hashtagList), set(mentionList)))
        H_H=[(x) for x in itertools.permutations(set(hashtagList),2)]
        M_M=[(x) for x in itertools.permutations(set(mentionList),2)]

        edges_all=K_K+M_K+H_K+H_H+H_M+M_M
        nodeList = hashtagList + mentionList + keywordList

        layer_files={}
        bipartite_files={}
        if len(keywordList)>1:
            k_k = pd.DataFrame(K_K);
            layer_files['K']=k_k
        elif len(keywordList)==1:
            K_K = keywordList
            k_k = pd.DataFrame(K_K);
            layer_files['K']=k_k


        if len(set(hashtagList))>1:
            h_h = pd.DataFrame(H_H);
            layer_files['H']=h_h

        elif len(set(hashtagList))==1:
            H_H=hashtagList
            h_h = pd.DataFrame(H_H);
            layer_files['H']=h_h


        if len(set(mentionList))>1:
            m_m = pd.DataFrame(M_M);
            layer_files['M']=m_m

        elif len(set(mentionList))==1:
            M_M=mentionList
            m_m = pd.DataFrame(M_M);
            layer_files['M']=m_m

        if len(hashtagList)>0 and len(mentionList)>0 and len(keywordList)>0:
            if len(H_K)==0 or len(hashtagList)<2:
                H_K = list(itertools.product(set(hashtagList),set(keywordList)))
            if len(M_K)==0 or len(mentionList)<2:
                M_K = list(itertools.product(set(mentionList),set(keywordList)))
            h_m = pd.DataFrame(H_M);
            m_k = pd.DataFrame(M_K);
            h_k = pd.DataFrame(H_K);
            bipartite_files = {"H-M": h_m, "M-K": m_k, "H-K": h_k}

        elif len(hashtagList)>0 and len(keywordList)>0:
            if len(H_K)==0  or len(hashtagList)<2:
                H_K = list(itertools.product(set(hashtagList),set(keywordList)))
            h_k = pd.DataFrame(H_K);
            bipartite_files = {"H-K": h_k}

        elif len(mentionList)>0 and len(keywordList)>0:
            if len(M_K)==0  or len(mentionList)<2:
                M_K = list(itertools.product(set(mentionList),set(keywordList)))
            m_k = pd.DataFrame(M_K);
            bipartite_files = {"M-K": m_k}

        elif len(mentionList)>0 and len(hashtagList)>0:
            h_m = pd.DataFrame(H_M);
            bipartite_files = {"H-M": h_m}

        def create_biased_rw(xx):
            (layer_files,bipartite_files,f)=xx
            print(f)
            if generate_tweet_from_graph:
                start = time.time()
                ''' Map the tweets from original stored graph using exact node names '''
                if (len(list(layer_files.keys()))) == 1:
                    walk_df = simulate_normal_walk(layer_files)
                else:
                    walk_df = simulate_walks(layer_files, bipartite_files)
                finish = time.time()
                display_time_taken(finish - start) 
                
                for i in walk_df['walk']:
                    for w in i:
                        walk=w.strip().split()
                        if len(walk)>5:
                            #print(walk,'\n')
                            f.write(tid+'\t')
                            for node in walk:
                                f.write(node+' ')
                        f.write('\n')
                f.write('\n')
                            # print('Biased RW write completed')

        create_biased_rw((layer_files,bipartite_files,preprocess_corpus_startnode))   
    tids+=1

       

preprocess_corpus_startnode.close()
