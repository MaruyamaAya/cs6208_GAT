import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

def normalize_features(matrix):
    row_sum = np.array(matrix.sum(1))
    row_inv = np.power(row_sum, -1).flatten()
    row_inv[np.isinf(row_inv)] = 0.
    row_mat_inv = sp.diags(row_inv)
    return row_mat_inv.dot(matrix)

def normalize_adjacency_matrix(matrix):
    row_sum = np.array(matrix.sum(1))
    row_inv_sqrt = np.power(row_sum, -0.5).flatten()
    row_inv_sqrt[np.isinf(row_inv_sqrt)] = 0.
    row_mat_inv_sqrt = sp.diags(row_inv_sqrt)
    return matrix.dot(row_mat_inv_sqrt).transpose().dot(row_mat_inv_sqrt)

def load_dataset(file_path):
    idx_features_labels = np.genfromtxt("{}cora.content".format(file_path), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = pd.get_dummies(idx_features_labels[:, -1]).values
    indices = np.array(idx_features_labels[:, 0], dtype=np.int32)
    index_map = {j: i for i, j in enumerate(indices)}
    unordered_edges = np.genfromtxt("{}cora.cites".format(file_path), dtype=np.int32)
    edges = np.array(list(map(index_map.get, unordered_edges.flatten())), dtype=np.int32).reshape(
        unordered_edges.shape)
    adjacency_matrix = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T.multiply(adjacency_matrix.T > adjacency_matrix) - \
                       adjacency_matrix.multiply(adjacency_matrix.T > adjacency_matrix)
    features = normalize_features(features)
    adjacency_matrix = normalize_adjacency_matrix(adjacency_matrix + sp.eye(adjacency_matrix.shape[0]))
    train_indices = range(199)
    val_indices = range(200, 500)
    test_indices = range(500, 1500)
    adjacency_matrix = torch.FloatTensor(np.array(adjacency_matrix.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    train_indices = torch.LongTensor(train_indices)
    val_indices = torch.LongTensor(val_indices)
    test_indices = torch.LongTensor(test_indices)
    return adjacency_matrix, features, labels, train_indices, val_indices, test_indices
