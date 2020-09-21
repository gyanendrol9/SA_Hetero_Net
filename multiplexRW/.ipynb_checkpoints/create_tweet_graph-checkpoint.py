import networkx as nx
import time

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

def create_tweet_graph(nodeList):
    start = time.time()

    ''' Give your input here in form of nodelist that matches with the exact spelling of nodes in the original graph'''
    ''' Example:~
        hashtagList = ["H_pakistan", "H_uriattack", "H_modi", "H_trumpwon"]
        mentionList = ["M_firstpost", "M_un"]
        keywordList = ["not", "single", "proof", "gvn", "asking", "for", "international", "inquiry", "but", "govt",
                       "refusing", "weird."]
        nodeList = hashtagList + mentionList + keywordList
    '''

    ''' Read stored attributed graph generated from twitter data '''
    H = nx.read_gpickle("corpus_graph.gpickle")
    nodeIndexList = [H.graph['node_to_id'].index(n) for n in nodeList]

    ''' Extract tweet subgraph from the original graph based on node information 
    & get most influential node of that subgraph'''
    S = H.subgraph(nodeIndexList)
    max_infl = -1; max_idx = -1
    for i, v in list(S.nodes(data=True)):
        # print(v['node_influence'])
        if (max_infl < v['node_influence']):
            max_infl = v['node_influence']
            max_idx = i
    print("Most influential node of the input tweet subgraph is: ", max_idx, S.nodes[max_idx], max_infl)

    start_node = max_idx; added_edge_list = list(); N = S.copy(); max_depth = 2
    for tgt_idx, tgt_data in H.nodes(data=True):
        if tgt_data['node_influence'] >= max_infl and tgt_idx != max_idx:
            try:
                n = nx.shortest_path_length(H, source=max_idx, target=tgt_idx)
                if (n <= max_depth):
                    print("New influential nodes added: ", tgt_idx, H.nodes[tgt_idx], n)
                    all_paths = [p for p in nx.all_shortest_paths(H, source=max_idx, target=tgt_idx)]
                    for path in all_paths:
                        for i in range(len(path) - 1):
                            N.add_edge(path[i], path[i + 1], weight=H[path[i]][path[i + 1]][0]['weight'])
                            print("New edge: ", (path[i], path[i + 1]), N[path[i]][path[i + 1]])
                            attr_dict = H.nodes[path[i + 1]]
                            N.add_node(path[i + 1], **attr_dict)
                            added_edge_list.append((H.nodes[max_idx], H.nodes[tgt_idx], H[path[i]][path[i + 1]][0]['weight']))
            except nx.NetworkXNoPath:
                print('No path')
                continue

    #nx.write_gpickle(S, "original_tweet_graph.gpickle")
    #nx.write_gpickle(N, "augmented_tweet_graph.gpickle")

    finish = time.time()
    display_time_taken(finish - start)

    return S, N, added_edge_list


