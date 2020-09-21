import networkx as nx
import pandas as pd
import time
from multiplexRW import * 
import random

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
            l = '\t'.join(str(h.pool_of_nodes[i]) for i in walk)
        else:
            l = '\t'.join(str(tweet_nodeList[i]) for i in walk)
        walk_list.append(l)
    walk_str = '\n'.join(str(w) for w in walk_list)
    walk_str += "\n"

    return walk_str, walk_list


def simulate_walks(layer_files, bipartite_files):
    multiplex_dict = dict.fromkeys(layer_files)
    for key, value in layer_files.items():
        multiplex_dict[key] = multilayer({key: value})
        multiplex_dict[key].dump_info()

    global h, H
    # h = hetero_multilayer(multiplex_dict, bipartite_files)
    # H = nx.from_scipy_sparse_matrix(h.supra_transition_matrix, edge_attribute='weight', create_using=nx.MultiDiGraph())

    X, Z = MultiRank(h)
    f=open("./Tweets_biased_multiplex_walk","a")
    for k, _ in Z.items():
        preprocess_transition_probs(X[k], Z[k])

        node_list = H.nodes()
        print("Printing the Walks!\n")
        for i in node_list:
            walk_str, walk_list = walks(i)
            print(walk_str);
            f.write(walk_str)
    f.close()  


if __name__ == '__main__':
    start = time.time()

    h_h = pd.read_csv("data/cooccurance_hashtags.csv", sep=",", names=["source", "target", "weight"])
    m_m = pd.read_csv("data/cooccurance_mentions.csv", sep=",", names=["source", "target", "weight"])
    k_k = pd.read_csv("data/cooccurance_keywords.csv", sep=",", names=["source", "target", "weight"])
    h_m = pd.read_csv("data/hashtags_mentions.csv", sep=",", names=["source", "target", "weight"])
    k_m = pd.read_csv("data/keywords_mentions.csv", sep=",", names=["source", "target", "weight"])
    k_h = pd.read_csv("data/keywords_hashtags.csv", sep=",", names=["source", "target", "weight"])
    m_k = pd.read_csv("data/mentions_keywords.csv", sep=",", names=["source", "target", "weight"])
    h_k = pd.read_csv("data/hashtags_keywords.csv", sep=",", names=["source", "target", "weight"])
    k_k.loc[k_k['source'].isin(stopwords), 'weight'] = 1
    k_k.loc[k_k['target'].isin(stopwords), 'weight'] = 1
    layer_files = {"H": h_h, "M": m_m, "K": k_k}
    bipartite_files = {"H-M": h_m, "K-M": k_m, "K-H": k_h, "M-K": m_k, "H-K": h_k}

    multiplex_dict = dict.fromkeys(layer_files)
    for key, value in layer_files.items():
        multiplex_dict[key] = multilayer({key: value})
        multiplex_dict[key].dump_info()

    global h, H
    h = hetero_multilayer(multiplex_dict, bipartite_files)
    H = nx.from_scipy_sparse_matrix(h.supra_transition_matrix, edge_attribute='weight', create_using=nx.MultiDiGraph())

    X, Z = MultiRank(h)
    generate_attributed_graph_from_corpus(X, Z)
    if elite:
        nx.write_gpickle(H, "data/corpus_graph_elite3.gpickle")
    else:
        nx.write_gpickle(H, "data/corpus_graph_influence3.gpickle")
    ''' Check if attributed graph created successfully '''
    # H = nx.read_gpickle("corpus_graph_influence.gpickle")
    # print(H.graph)
    
    walk_df = simulate_walks(layer_files, bipartite_files)

    finish = time.time()
    display_time_taken(finish - start)
