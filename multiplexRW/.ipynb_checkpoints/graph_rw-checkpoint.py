import networkx as nx
import numpy as np
import pandas as pd
import time
import itertools
from networks import multilayer, hetero_multilayer
from centrality import *
from config import *
from create_tweet_graph import create_tweet_graph
import random
from collections import OrderedDict
from decimal import Decimal

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
    unnormalized_probs = []
    for dst_nbr in sorted(H.neighbors(dst)):
        if dst_nbr == src:
            unnormalized_probs.append(H[dst][dst_nbr][0]['weight'] / p if 0 in H[dst][dst_nbr] else H[dst][dst_nbr]['weight']/p)
        elif H.has_edge(dst_nbr, src):
            unnormalized_probs.append(H[dst][dst_nbr][0]['weight'] if 0 in H[dst][dst_nbr] else H[dst][dst_nbr]['weight'])
        else:
            unnormalized_probs.append(H[dst][dst_nbr][0]['weight'] / q if 0 in H[dst][dst_nbr] else H[dst][dst_nbr]['weight'] / q)
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

def preprocess_transition_probs(X=None, Z=None):
    global alias_nodes, alias_edges
    alias_nodes = {}

    if X is not None and Z is not None:
        sort_keys = sorted(h.multiplexes.keys())
        if (len(sort_keys)) == 2:
            k1 = sort_keys[0]; n1 = h.node_num_dict[k1]
            k2 = sort_keys[1]; n2 = h.node_num_dict[k2]
            X[0:n1] = np.multiply(Z[0], X[0:n1])
            X[n1:n1+n2] = np.multiply(Z[1], X[n1:n1 + n2])
        elif (len(sort_keys)) == 3:
            k1 = sort_keys[0]; n1 = h.node_num_dict[k1]
            k2 = sort_keys[1]; n2 = h.node_num_dict[k2]
            k3 = sort_keys[2]; n3 = h.node_num_dict[k3]
            X[0:n1] =  np.multiply(Z[0], X[0:n1])
            X[n1:n1 + n2] = np.multiply(Z[1], X[n1:n1 + n2])
            X[n1+n2:] =  np.multiply(Z[2], X[n1+n2:])
        X = np.divide(np.array(X), np.array(X).sum())

    for node in H.nodes():
        unnormalized_probs = [r * H[node][nbr][0]['weight'] if 0 in H[node][nbr] else H[node][nbr]['weight'] for nbr in sorted(H.neighbors(node))]
        if X is not None and Z is not None:
            restart_probs = [(1-r) * X[nbr] for nbr in sorted(H.neighbors(node))]
            final_probs =  sum(restart_probs, unnormalized_probs)
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

def walks(start_node):
    walk_list = list()
    for walk_iter in range(num_walks):
        source = start_node
        walk = list(); walk.append(source)
        for w_length in range(walk_length):
            cur = walk[-1]
            cur_nbrs = sorted(H.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break
        l = '\t'.join(str(h.pool_of_nodes[i]) for i in walk)
        walk_list.append(l)
    walk_str = '\n'.join(str(w) for w in walk_list)
    walk_str += "\n"

    for line in walk_str:
        f.write(line)

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
            #print(walk_str); print()

            d = OrderedDict()
            d['gamma'] = round(k * 0.1, 3)
            d['layer_centrality'] = ['{:.2e}'.format(Decimal(itm)) for itm in Z[k]]
            d['node_centrality'] =  ['{:.2e}'.format(Decimal(itm)) for itm in X[k]]
            d['p'] =  p
            d['q'] = q
            d['num_walks'] = num_walks,
            d['walk_length'] = walk_length
            d['restart'] = round(r, 2)
            d['jump'] = delta
            d['walk'] = walk_list
            walk_df.append(d)

    walk_df = pd.DataFrame(walk_df)
    walk_df.to_csv("walks.csv", index=False)

    return walk_df

if __name__ == '__main__':
    start = time.time()
    global f
    f = open("walks.txt", 'w')

    h_h = pd.read_csv("data/cooccured_hashtags.csv", sep=",", names=["source", "target", "weight"])
    m_m = pd.read_csv("data/cooccured_mentions.csv", sep=",", names=["source", "target", "weight"])
    k_k = pd.read_csv("data/keywords_directed.csv", sep=",", names=["source", "target", "weight"])
    h_m = pd.read_csv("data/hashtags_mentions.csv", sep=",", names=["source", "target", "weight"])
    k_m = pd.read_csv("data/keywords_mentions.csv", sep=",", names=["source", "target", "weight"])
    k_h = pd.read_csv("data/keywords_hashtags.csv", sep=",", names=["source", "target", "weight"])

    layer_files = {"H": h_h, "M": m_m, "K": k_k}
    bipartite_files = {"H-M": h_m, "K-M": k_m, "K-H": k_h}

    walk_df = simulate_walks(layer_files, bipartite_files)

    f.close()
    finish = time.time()
    display_time_taken(finish - start)

