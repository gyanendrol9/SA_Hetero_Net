import sys
import numpy as np
import scipy
import scipy.sparse as sp
import networkx as nx
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
from scipy.sparse import identity, spdiags, linalg
from networkx.algorithms import bipartite
import config as cnf
import pandas as pd

class hetero_multilayer:
    def __init__(self, layer_multiplexes, bipartite_files=None):
        #TODO check and verify inputs
        self.multiplexes = layer_multiplexes
        print("\nGenerating bipartite matrix...")
        if(len(layer_multiplexes.keys())==1 or bipartite_files is None) :
            k = list(layer_multiplexes.keys())
            self.total_nodes = layer_multiplexes[k[0]].num_nodes
            self.num_layers = 1
            self.pool_of_nodes = list(layer_multiplexes[k[0]].pool_of_nodes)
            self.supra_adjacency_matrix = layer_multiplexes[k[0]].layers[0].adj_matrix
            self.normed_supra_adjacency_matrix = layer_multiplexes[k[0]].layers[0].normed_adj_matrix
            self.supra_transition_matrix = layer_multiplexes[k[0]].layers[0].normed_adj_matrix

        else:
            self.bipartite_matrix = dict.fromkeys(bipartite_files, None)
            self.bipartite_G = dict.fromkeys(bipartite_files, None)
            for key, value in bipartite_files.items():
                if (len(value.columns)) == 2:
                    value.columns = ["source", "target"]
                    value['weight'] = [1.0] * value.shape[0]
                elif (len(value.columns)) == 3:
                    value.columns = ["source", "target", "weight"]
                bipartite_rel = value
                self.bipartite_G[key], self.bipartite_matrix[key] = self.get_bipartite_graph(layer_multiplexes[key.split("-")[0]], layer_multiplexes[key.split("-")[1]], bipartite_rel)
            print("Expanding bipartite matrix to fit the multiplex network...")
            self.supra_adjacency_matrix = self.compute_adjacency_matrix(self.multiplexes, self.bipartite_matrix)
            with scipy.errstate(divide='ignore', invalid='ignore'):
                DI = spdiags(1.0 / scipy.array(self.supra_adjacency_matrix.sum(axis=1).flat), [0], self.total_nodes, self.total_nodes)
            self.normed_supra_adjacency_matrix = DI * self.supra_adjacency_matrix
            self.supra_transition_matrix = self.compute_transition_matrix()

    def get_bipartite_graph(self, S, T, bipartite_rel):
        cond_1 = all(elem in list(S.pool_of_nodes) for elem in list(bipartite_rel['source']))
        cond_2 = all(elem in list(T.pool_of_nodes) for elem in list(bipartite_rel['target']))
        e1 = [elem in list(S.pool_of_nodes) for elem in list(bipartite_rel['source'])]
        e2 = [elem in list(T.pool_of_nodes) for elem in list(bipartite_rel['target'])]
        if True:
        #if cond_1 and cond_2 :
            G = nx.Graph()
            G.add_nodes_from(S.pool_of_nodes, bipartite=0)
            G.add_nodes_from(T.pool_of_nodes, bipartite=1)
            G.add_weighted_edges_from([(row['source'], row['target'], row['weight']) for idx, row in bipartite_rel.iterrows()], weight='weight')
            G.remove_edges_from(nx.selfloop_edges(G))
            r = nx.get_node_attributes(G,'bipartite')
            B = bipartite.biadjacency_matrix(G, row_order=S.pool_of_nodes, column_order=T.pool_of_nodes)
            return G, B
        # else:
        #     print("Source & Target nodes don't match with Bipartite Relation nodes.")
        #     exit(-2)

    def compute_adjacency_matrix(self, multiplexes, bipartite_matrix):
        total_nodes = 0; total_edges = 0; p_of_nodes = list()
        node_dict = dict.fromkeys(multiplexes, 0)
        matrix_dict = dict.fromkeys(multiplexes, 0)
        bp_matrix_dict = dict.fromkeys(bipartite_matrix, None)
        bp_keys = bipartite_matrix.keys()

        for key in sorted(multiplexes.keys()):
            nodes = multiplexes[key].num_nodes
            pool_of_nodes = multiplexes[key].layers[0].pool_of_nodes
            total_nodes += nodes
            p_of_nodes += list(pool_of_nodes)
            node_dict[key] = nodes
            matrix_dict[key] = multiplexes[key].layers[0].adj_matrix

            for b in bp_keys:
                if key == b[0]:
                    if bp_matrix_dict[b] is None or bp_matrix_dict[b[::-1]] is None :
                        bp_matrix_dict[b] = bipartite_matrix[b]
                        bp_matrix_dict[b[::-1]] = bipartite_matrix[b].transpose()

        self.total_nodes = total_nodes
        L = len(multiplexes.keys())
        self.num_layers = L
        self.pool_of_nodes = p_of_nodes
        self.node_num_dict = node_dict
        self.bipartite_matrix = bp_matrix_dict
        supra_adjacency_matrix = lil_matrix((total_nodes, total_nodes), dtype=np.float)
        sort_keys = sorted(multiplexes.keys())

        if (len(sort_keys)) == 1:
            print("No bipartite relation possible!!")
            exit(-2)

        elif (len(sort_keys)) == 2:
            k1 = sort_keys[0]; n1 = node_dict[k1]
            k2 = sort_keys[1]; n2 = node_dict[k2]
            supra_adjacency_matrix[0:n1, 0:n1] = matrix_dict[k1]
            if k1+"-"+k2 in bp_matrix_dict :
                if bp_matrix_dict[k1+"-"+k2] is not None:
                    supra_adjacency_matrix[0:n1, n1:n1 + n2] = bp_matrix_dict[k1+"-"+k2]
            if k2 + "-" + k1 in bp_matrix_dict:
                if bp_matrix_dict[k2+"-"+k1] is not None:
                    supra_adjacency_matrix[n1:n1+n2, 0:n1] = bp_matrix_dict[k2+"-"+k1]
            supra_adjacency_matrix[n1:n1+n2, n1:n1+n2] = matrix_dict[k2]

        elif (len(sort_keys)) == 3:
            k1 = sort_keys[0]; n1 = node_dict[k1]
            k2 = sort_keys[1]; n2 = node_dict[k2]
            k3 = sort_keys[2]; n3 = node_dict[k3]
            supra_adjacency_matrix[0:n1, 0:n1] = matrix_dict[k1]
            if k1 + "-" + k2 in bp_matrix_dict:
                if bp_matrix_dict[k1+"-"+k2] is not None:
                    supra_adjacency_matrix[0:n1, n1:n1 + n2] = bp_matrix_dict[k1+"-"+k2]
            if k1 + "-" + k3 in bp_matrix_dict:
                if bp_matrix_dict[k1+"-"+k3] is not None:
                    supra_adjacency_matrix[0:n1, n1+n2 : n1+n2+n3] = bp_matrix_dict[k1+"-"+k3]

            if k2 + "-" + k1 in bp_matrix_dict:
                if bp_matrix_dict[k2 + "-" + k1] is not None:
                    supra_adjacency_matrix[n1:n1 + n2, 0:n1] = bp_matrix_dict[k2 + "-" + k1]
            supra_adjacency_matrix[n1:n1 + n2, n1:n1 + n2] = matrix_dict[k2]
            if k2 + "-" + k3 in bp_matrix_dict:
                if bp_matrix_dict[k2 + "-" + k3] is not None:
                    supra_adjacency_matrix[n1:n1 + n2, n1 + n2:n1 + n2 + n3] = bp_matrix_dict[k2 + "-" + k3]

            if k3 + "-" + k1 in bp_matrix_dict:
                if bp_matrix_dict[k3 + "-" + k1] is not None:
                    supra_adjacency_matrix[n1+n2:n1+n2+n3, 0:n1] = bp_matrix_dict[k3 + "-" + k1]
            if k3 + "-" + k2 in bp_matrix_dict:
                if bp_matrix_dict[k3 + "-" + k2] is not None:
                    supra_adjacency_matrix[n1+n2:n1+n2+n3, n1:n1 + n2] = bp_matrix_dict[k3 + "-" + k2]
            supra_adjacency_matrix[n1+n2:n1+n2+n3, n1+n2:n1+n2+n3] = matrix_dict[k3]

        else:
            print("Invalid number of layers!!")
            exit(-2)

        return(supra_adjacency_matrix)

    def compute_transition_matrix(self):
        total_edges = 0
        if cnf.compute_weights:
            for key in self.multiplexes.keys():
                total_edges += self.multiplexes[key].layers[0].edge_df.shape[0]
            for key in self.multiplexes.keys():
                cnf.delta[key] = round(self.multiplexes[key].layers[0].edge_df.shape[0]/ total_edges, 4)

        L = len(self.multiplexes.keys())
        self.supra_transition_matrix = self.supra_adjacency_matrix
        sort_keys = sorted(self.multiplexes.keys())
        if (len(sort_keys)) == 1:
            print("No bipartite relation possible!!")
            exit(-2)
        elif (len(sort_keys)) == 2:
            k1 = sort_keys[0]; n1 = self.node_num_dict[k1]
            k2 = sort_keys[1]; n2 = self.node_num_dict[k2]
            self.supra_transition_matrix[0:n1, 0:n1] = cnf.delta[k1] * self.supra_transition_matrix[0:n1, 0:n1]
            if k1+"-"+k2 in self.bipartite_matrix :
                if self.bipartite_matrix[k1+"-"+k2] is not None:
                    self.supra_transition_matrix[0:n1, n1:n1 + n2] = (1.0 - cnf.delta[k1]) / (
                            L - 1) * 1.0 * self.supra_transition_matrix[0:n1, n1:n1 + n2]
            if k2 + "-" + k1 in self.bipartite_matrix:
                if self.bipartite_matrix[k2+"-"+k1] is not None:
                    self.supra_transition_matrix[n1:n1+n2, 0:n1] = (1.0-cnf.delta[k2])/(L-1) * 1.0 * self.supra_transition_matrix[n1:n1+n2, 0:n1]
            self.supra_transition_matrix[n1:n1+n2, n1:n1+n2] = cnf.delta[k2] * self.supra_transition_matrix[n1:n1+n2, n1:n1+n2]

        elif (len(sort_keys)) == 3:
            k1 = sort_keys[0]; n1 = self.node_num_dict[k1]
            k2 = sort_keys[1]; n2 = self.node_num_dict[k2]
            k3 = sort_keys[2]; n3 = self.node_num_dict[k3]
            self.supra_transition_matrix[0:n1, 0:n1] = cnf.delta[k1] * self.supra_transition_matrix[0:n1, 0:n1]
            if k1 + "-" + k2 in self.bipartite_matrix:
                if self.bipartite_matrix[k1+"-"+k2] is not None:
                    self.supra_transition_matrix[0:n1, n1:n1 + n2] = (1.0 - cnf.delta[k1]) / (
                            L - 1) * 1.0 * self.supra_transition_matrix[0:n1, n1:n1 + n2]
            if k1 + "-" + k3 in self.bipartite_matrix:
                if self.bipartite_matrix[k1+"-"+k3] is not None:
                    self.supra_transition_matrix[0:n1, n1+n2 : n1+n2+n3] = (1.0 - cnf.delta[k1]) / (
                        L - 1) * 1.0 * self.supra_transition_matrix[0:n1, n1+n2 : n1+n2+n3]

            if k2 + "-" + k1 in self.bipartite_matrix:
                if self.bipartite_matrix[k2 + "-" + k1] is not None:
                    self.supra_transition_matrix[n1:n1 + n2, 0:n1] = (1.0-cnf.delta[k2])/(L-1) * 1.0 * self.supra_transition_matrix[n1:n1 + n2, 0:n1]
            self.supra_transition_matrix[n1:n1 + n2, n1:n1 + n2] = cnf.delta[k2] * self.supra_transition_matrix[n1:n1 + n2, n1:n1 + n2]
            if k2 + "-" + k3 in self.bipartite_matrix:
                if self.bipartite_matrix[k2 + "-" + k3] is not None:
                    self.supra_transition_matrix[n1:n1 + n2, n1 + n2:n1 + n2 + n3] = (1.0-cnf.delta[k2])/(L-1) * 1.0 * self.supra_transition_matrix[n1:n1 + n2, n1 + n2:n1 + n2 + n3]

            if k3 + "-" + k1 in self.bipartite_matrix:
                if self.bipartite_matrix[k3 + "-" + k1] is not None:
                    self.supra_transition_matrix[n1+n2:n1+n2+n3, 0:n1] = (1.0-cnf.delta[k3])/(L-1) * 1.0 * self.supra_transition_matrix[n1+n2:n1+n2+n3, 0:n1]
            if k3 + "-" + k2 in self.bipartite_matrix:
                if self.bipartite_matrix[k3 + "-" + k2] is not None:
                    self.supra_transition_matrix[n1+n2:n1+n2+n3, n1:n1 + n2] = (1.0-cnf.delta[k3])/(L-1) * 1.0 * self.supra_transition_matrix[n1+n2:n1+n2+n3, n1:n1 + n2]
            self.supra_transition_matrix[n1+n2:n1+n2+n3, n1+n2:n1+n2+n3] = cnf.delta[k3] * self.supra_transition_matrix[n1+n2:n1+n2+n3, n1+n2:n1+n2+n3]

        else:
            print("Invalid number of layers!!")
            exit(-2)
        with scipy.errstate(divide='ignore', invalid='ignore'):
            DI = spdiags(1.0 / scipy.array(self.supra_transition_matrix.sum(axis=1).flat), [0], self.total_nodes, self.total_nodes)
        self.supra_transition_matrix =  DI * self.supra_transition_matrix

        return(self.supra_transition_matrix)

class multilayer:
    def __init__(self, multiplex_dict):
        self.layers = list()
        self.relations = list(multiplex_dict.keys())
        self.num_layers = len(list(multiplex_dict.keys()))
        self.num_nodes = 0
        self.pool_of_nodes = list()

        for key, layer_file in multiplex_dict.items():
            A = layer(key, layer_file)
            self.layers.append(A)

        self.pool_of_nodes = list(self.layers[0].pool_of_nodes)

        # for A in self.layers:
        #     self.pool_of_nodes = self.pool_of_nodes + list(A.pool_of_nodes)
        # self.pool_of_nodes = set(np.sort(self.pool_of_nodes))
        self.num_nodes = len(self.pool_of_nodes)

    def dump_info(self):
        i = 0
        for l in self.layers:
            sys.stderr.write("--------\nLayer: %d\n" % i)
            l.dump_info()
            i += 1

class layer:
    def __init__(self, rel, layer_dict):
        self.relation_id = rel
        self.num_nodes = 0
        self.num_layers = -1 # in case of multiplex of multiplexes
        self.pool_of_nodes = None
        self.adj_matrix, self.normed_adj_matrix, self.laplacian = None, None, None
        self.edge_df = layer_dict

        if (len(self.edge_df.columns)) == 1:
            self.edge_df.columns = ["source"]
            self.edge_df['target'] = self.edge_df['source']
            self.edge_df['weight'] = [1.0] * self.edge_df.shape[0]
        elif(len(self.edge_df.columns)) == 2:
            self.edge_df.columns = ["source", "target"]
            self.edge_df['weight'] = [1.0] * self.edge_df.shape[0]
        elif(len(self.edge_df.columns)) == 3:
            self.edge_df.columns = ["source", "target", "weight"]

        self.pool_of_nodes = set(np.sort(list(self.edge_df['source'].unique()) + list(self.edge_df['target'].unique())))
        self.num_nodes = len(self.pool_of_nodes)
        if rel == "K":
            self.G = nx.from_pandas_edgelist(self.edge_df, edge_attr=True, create_using=nx.DiGraph())
        else:
            self.G = nx.from_pandas_edgelist(self.edge_df, edge_attr=True, create_using=nx.Graph())

        self.G.remove_edges_from(nx.selfloop_edges(self.G))
        self.make_matrices(self.G)

    def make_matrices(self, G):
        self.adj_matrix = nx.to_scipy_sparse_matrix(G, nodelist=self.pool_of_nodes)
        n, m = self.adj_matrix.shape
        D = self.adj_matrix.sum(axis=1).flatten()
        D = scipy.sparse.spdiags(D, [0], n, m, format='csr')
        self.laplacian = csr_matrix(D - self.adj_matrix)
        with scipy.errstate(divide='ignore', invalid='ignore'):
            DI = spdiags(1.0 / scipy.array(self.adj_matrix.sum(axis=1).flat), [0], n, m)
        self.normed_adj_matrix = DI * self.adj_matrix

    def dump_info(self):
        N, M = self.adj_matrix.shape
        K = self.adj_matrix.nnz
        sys.stderr.write("Layer File: %s\nNodes: %d Edges: %d\n" % (self.relation_id, N, K))

