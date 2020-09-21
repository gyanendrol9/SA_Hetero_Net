import networkx as nx
import pandas as pd
import time
from networks import multilayer, hetero_multilayer
from centrality import *
from config import *

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

def generate_attributed_graph_from_corpus(X=None, Z=None):
    pool_of_nodes = list(h.pool_of_nodes)

    for node in pool_of_nodes:
        H.nodes[pool_of_nodes.index(node)]['node_id'] = node
        H.nodes[pool_of_nodes.index(node)]['layer_id'] = node[0].upper() if node[0].upper() in ['H', 'M'] and '_' in \
                                                                            node[1] else 'K'
        if X is not None:
            H.nodes[pool_of_nodes.index(node)]['node_influence'] = X[int(choice_of_gamma * 10)][
            pool_of_nodes.index(node)]

    H.graph['node_to_id'] = h.pool_of_nodes
    H.graph['layer_to_id'] = sorted(h.multiplexes.keys())
    if Z is not None:
        H.graph['layer_influence'] = Z[int(choice_of_gamma * 10)]

    # print(H.nodes(data=True))
    # print(H.edges(data=True))
    # print(H.graph)


if __name__ == '__main__':
    start = time.time()


    h_h = pd.read_csv("data/cooccured_hashtags.csv", sep=",", names=["source", "target", "weight"])
    m_m = pd.read_csv("data/cooccured_mentions.csv", sep=",", names=["source", "target", "weight"])
    k_k = pd.read_csv("data/keywords_directed.csv", sep=",", names=["source", "target", "weight"])
    h_m = pd.read_csv("data/hashtags_mentions.csv", sep=",", names=["source", "target", "weight"])
    k_m = pd.read_csv("data/keywords_mentions.csv", sep=",", names=["source", "target", "weight"])
    k_h = pd.read_csv("data/keywords_hashtags.csv", sep=",", names=["source", "target", "weight"])

    layer_files = {"H": h_h, "M": m_m, "K": k_k}
    bipartite_files = {"H-M": h_m, "K-M": k_m, "K-H": k_h}

    multiplex_dict = dict.fromkeys(layer_files)
    for key, value in layer_files.items():
        multiplex_dict[key] = multilayer({key: value})
        multiplex_dict[key].dump_info()

    global h, H
    h = hetero_multilayer(multiplex_dict, bipartite_files)
    H = nx.from_scipy_sparse_matrix(h.supra_transition_matrix, edge_attribute='weight', create_using=nx.MultiDiGraph())

    X, Z = MultiRank(h)
    generate_attributed_graph_from_corpus(X, Z)
    nx.write_gpickle(H, "corpus_graph.gpickle")
    ''' Check if attributed graph created successfully '''
    # H = nx.read_gpickle("corpus_graph.gpickle")
    # print(H.graph)

    finish = time.time()
    display_time_taken(finish - start)




