import math
import numpy as np
import scipy
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
from scipy.sparse import identity, spdiags, linalg
from scipy.linalg import norm
from config import *

def MultiRank_Nodes_Layers(H, alpha, gamma, s, a):
    v_quadratic_error = 0.001

    z = np.ones(H.num_layers, )
    g = H.supra_adjacency_matrix
    #g = lil_matrix((H.total_nodes, H.total_nodes), dtype=np.float) # Without bipartite connection
    sort_keys = sorted(H.multiplexes.keys())
    if (len(sort_keys)) == 2:
        k1 = sort_keys[0]; n1 = H.node_num_dict[k1]
        k2 = sort_keys[1]; n2 = H.node_num_dict[k2]
        g[0:n1, 0:n1] = H.supra_adjacency_matrix[0:n1, 0:n1].T * z[0]
        g[n1:n1 + n2, n1:n1 + n2] = H.supra_adjacency_matrix[n1:n1 + n2, n1:n1 + n2].T * z[1]
    elif (len(sort_keys)) == 3:
        k1 = sort_keys[0]; n1 = H.node_num_dict[k1]
        k2 = sort_keys[1]; n2 = H.node_num_dict[k2]
        k3 = sort_keys[2]; n3 = H.node_num_dict[k3]
        g[0:n1, 0:n1] = H.supra_adjacency_matrix[0:n1, 0:n1].T * z[0]
        g[n1:n1 + n2, n1:n1 + n2] = H.supra_adjacency_matrix[n1:n1 + n2, n1:n1 + n2].T * z[1]
        g[n1 + n2: n1 + n2 + n3, n1 + n2: n1 + n2 + n3] = H.supra_adjacency_matrix[n1 + n2: n1 + n2 + n3,
                                                          n1 + n2: n1 + n2 + n3].T * z[2]

    B_in = lil_matrix((H.num_layers, H.total_nodes), dtype=np.float)
    W = np.zeros(H.num_layers)

    if (len(sort_keys)) == 2:
        k1 = sort_keys[0]; n1 = H.node_num_dict[k1]
        k2 = sort_keys[1]; n2 = H.node_num_dict[k2]
        W[0] = H.multiplexes[k1].layers[0].adj_matrix.sum()
        W[1] = H.multiplexes[k2].layers[0].adj_matrix.sum()
        if k1 + "-" + k2 in H.bipartite_matrix:
            if H.bipartite_matrix[k1 + "-" + k2] is not None:
                tmp = H.bipartite_matrix[k1 + "-" + k2].T.sum(axis=0).ravel()
                with scipy.errstate(divide='ignore', invalid='ignore'):
                    n_tmp = np.array(tmp / W[0])
                    n_tmp[n_tmp == np.inf] = 0;
                    n_tmp[np.where(np.isnan(n_tmp))] = 0;
                    B_in[0:1, 0:n1] = lil_matrix(n_tmp)
        if k2 + "-" + k1 in H.bipartite_matrix:
            if H.bipartite_matrix[k2 + "-" + k1] is not None:
                tmp = H.bipartite_matrix[k2 + "-" + k1].T.sum(axis=0).ravel()
                with scipy.errstate(divide='ignore', invalid='ignore'):
                    n_tmp = np.array(tmp / W[1])
                    n_tmp[n_tmp == np.inf] = 0;
                    n_tmp[np.where(np.isnan(n_tmp))] = 0;
                    B_in[1:, n1:] = lil_matrix(n_tmp)
    elif (len(sort_keys)) == 3:
        k1 = sort_keys[0]; n1 = H.node_num_dict[k1]
        k2 = sort_keys[1]; n2 = H.node_num_dict[k2]
        k3 = sort_keys[2]; n3 = H.node_num_dict[k3]
        W[0] = H.multiplexes[k1].layers[0].adj_matrix.sum()
        W[1] = H.multiplexes[k2].layers[0].adj_matrix.sum()
        W[2] = H.multiplexes[k3].layers[0].adj_matrix.sum()
        if k1 + "-" + k2 in H.bipartite_matrix:
            if H.bipartite_matrix[k1 + "-" + k2] is not None:
                tmp = H.bipartite_matrix[k1 + "-" + k2].T.sum(axis=0).ravel()
                B_in[0:1, 0:n1] = tmp
        if k1 + "-" + k3 in H.bipartite_matrix:
            if H.bipartite_matrix[k1 + "-" + k3] is not None:
                tmp = H.bipartite_matrix[k1 + "-" + k3].T.sum(axis=0).ravel()
                with scipy.errstate(divide='ignore', invalid='ignore'):
                    n_tmp = B_in[0:1, 0:n1] + tmp
                    n_tmp = np.array(n_tmp / W[0])
                    n_tmp [n_tmp==np.inf] = 0;
                    n_tmp[np.where(np.isnan(n_tmp))] = 0;
                    B_in[0:1, 0:n1] = lil_matrix(n_tmp)

        if k2 + "-" + k1 in H.bipartite_matrix:
            if H.bipartite_matrix[k2 + "-" + k1] is not None:
                tmp = H.bipartite_matrix[k2 + "-" + k1].T.sum(axis=0).ravel()
                B_in[1:2, n1:n1+n2] = tmp
        if k2 + "-" + k3 in H.bipartite_matrix:
            if H.bipartite_matrix[k2 + "-" + k3] is not None:
                tmp = H.bipartite_matrix[k2 + "-" + k3].T.sum(axis=0).ravel()
                with scipy.errstate(divide='ignore', invalid='ignore'):
                    n_tmp = B_in[1:2, n1:n1+n2] + tmp
                    n_tmp = np.array(n_tmp / W[1])
                    n_tmp[n_tmp == np.inf] = 0;
                    n_tmp[np.where(np.isnan(n_tmp))] = 0;
                    B_in[1:2, n1:n1+n2] = lil_matrix(n_tmp)

        if k3 + "-" + k1 in H.bipartite_matrix:
            if H.bipartite_matrix[k3 + "-" + k1] is not None:
                tmp = H.bipartite_matrix[k3 + "-" + k1].T.sum(axis=0).ravel()
                B_in[2:, n1+n2:] = tmp
        if k3 + "-" + k2 in H.bipartite_matrix:
            if H.bipartite_matrix[k3 + "-" + k2] is not None:
                tmp = H.bipartite_matrix[k3 + "-" + k2].T.sum(axis=0).ravel()
                with scipy.errstate(divide='ignore', invalid='ignore'):
                    n_tmp = B_in[2:, n1 + n2:] + tmp
                    n_tmp = np.array(n_tmp / W[2])
                    n_tmp[n_tmp == np.inf] = 0;
                    n_tmp[np.where(np.isnan(n_tmp))] = 0;
                    B_in[2:, n1 + n2:] = lil_matrix(n_tmp)

    D = g.sum(axis=1)
    D [D<1.0] = 1.0
    D = spdiags(1.0 / scipy.array(D.flat), [0], H.total_nodes, H.total_nodes)

    x0 = g.sum(axis=0) + g.sum(axis=1).T
    x0 = scipy.array(x0)
    with scipy.errstate(divide='ignore', invalid='ignore'):
        x0 = x0.T / np.count_nonzero(x0)
        x0[x0 == np.inf] = 0;
        x0[np.where(np.isnan(x0))] = 0;
        x0 = scipy.array(x0)

    l = scipy.array(g.sum(axis=0))
    jump = scipy.array(alpha * l.T)
    jump = np.divide(jump, jump.sum())
    x = x0
    x = g.dot(D).dot(np.multiply(x, jump)) + np.multiply(x, 1-jump).sum(axis=0) * x0
    x = np.divide(x, x.sum())

    z1 = np.power(B_in.sum(axis=1), a)
    z2 = B_in.dot(np.power(x, (s * gamma)))
    with scipy.errstate(divide='ignore', invalid='ignore'):
        n_tmp = z2
        n_tmp = np.array(n_tmp / B_in.sum(axis=1))
        n_tmp[n_tmp == np.inf] = 0;
        n_tmp[np.where(np.isnan(n_tmp))] = 0;
        z2 = n_tmp
    z = np.multiply(z1, (np.power(z2, s)))
    z = np.divide(z, z.sum())
    #normalized = (x - x.min()) / (x.max() - x.min())

    count = 0;
    last_x = np.ones(H.total_nodes, ) * np.inf
    while (True):
        last_x = x
        g = lil_matrix((H.total_nodes, H.total_nodes), dtype=np.float)
        sort_keys = sorted(H.multiplexes.keys())
        n_z = list()
        for item in z.tolist():
            n_z.append(item)
        z = np.array(n_z)
        z = z.reshape(H.num_layers, )

        if (len(sort_keys)) == 2:
            k1 = sort_keys[0]; n1 = H.node_num_dict[k1]
            k2 = sort_keys[1]; n2 = H.node_num_dict[k2]
            g[0:n1, 0:n1] = H.supra_adjacency_matrix[0:n1, 0:n1].T * z[0]
            g[n1:n1 + n2, n1:n1 + n2] = H.supra_adjacency_matrix[n1:n1 + n2, n1:n1 + n2].T * z[1]
        elif (len(sort_keys)) == 3:
            k1 = sort_keys[0]; n1 = H.node_num_dict[k1]
            k2 = sort_keys[1]; n2 = H.node_num_dict[k2]
            k3 = sort_keys[2]; n3 = H.node_num_dict[k3]
            g[0:n1, 0:n1] = H.supra_adjacency_matrix[0:n1, 0:n1].T * z[0]
            g[n1:n1 + n2, n1:n1 + n2] = H.supra_adjacency_matrix[n1:n1 + n2, n1:n1 + n2].T * z[1]
            g[n1 + n2: n1 + n2 + n3, n1 + n2: n1 + n2 + n3] = H.supra_adjacency_matrix[n1 + n2: n1 + n2 + n3,
                                                              n1 + n2: n1 + n2 + n3].T * z[2]

        D = g.sum(axis=1)
        D[D < 1.0] = 1.0
        D = spdiags(1.0 / scipy.array(D.flat), [0], H.total_nodes, H.total_nodes)

        x0 = g.sum(axis=0) + g.sum(axis=1).T
        with scipy.errstate(divide='ignore', invalid='ignore'):
            x0 = x0.T / np.count_nonzero(x0)
            x0[x0 == np.inf] = 0;
            x0[np.where(np.isnan(x0))] = 0;
        l = scipy.array(g.sum(axis=0))
        jump = scipy.array(alpha * l.T)
        jump = np.divide(jump, jump.sum())
        x = g.dot(D).dot(np.multiply(x, jump)) + np.multiply(np.multiply(x, 1 - jump).sum(axis=0), x0)
        x = np.divide(x, x.sum())

        z1 = np.power(B_in.sum(axis=1), a)
        z2 = B_in.dot(np.power(x, (s * gamma)))
        with scipy.errstate(divide='ignore', invalid='ignore'):
            n_tmp = z2
            n_tmp = np.array(n_tmp / B_in.sum(axis=1))
            n_tmp[n_tmp == np.inf] = 0;
            n_tmp[np.where(np.isnan(n_tmp))] = 0;
            z2 = n_tmp
        z = np.multiply(z1, (np.power(z2, s)))
        z = np.divide(z, z.sum())
        try:
            normed = norm(x - last_x)
        except:
            print(count)
            break

        if normed < v_quadratic_error:
            break
        elif(count>100):
            break
        count = count + 1

    return x,z

def MultiRank(H):
    s = 1; a = 0
    X = dict(); Z = dict()
    for ig in gamma_range:
        gamma = 0.1 * ig
        x, z = MultiRank_Nodes_Layers(H, r, gamma, s, a)
        X[ig]=[i for i in sorted(enumerate(x.tolist()), key=lambda x:x[1], reverse=True)]
        Z[ig]=[i for i in sorted(enumerate(z.tolist()), key=lambda x:x[1], reverse=True)]

    X_list = dict.fromkeys(X, [0] * H.total_nodes)
    Z_list = dict.fromkeys(X, [0] * H.num_layers)
    for ig in gamma_range:
        for k, v in Z.items():
            for item in v:
                layer_idx = item[0]
                influence = item[1][0]
                Z_list[ig][layer_idx] = influence
        Z_list[ig] = np.divide(np.array(Z_list[ig]), np.array(Z_list[ig]).sum())
        for k, v in X.items():
            for item in v:
                node_idx = item[0]
                influence = item[1][0]
                X_list[ig][node_idx] = influence
        X_list[ig] = np.divide(np.array(X_list[ig]), np.array(X_list[ig]).sum())

    return X_list, Z_list