import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys
import tensorflow.compat.v1 as tf
tf.config.run_functions_eagerly(True)
tf.disable_eager_execution()


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("/home/zihe-leon/Desktop/RobustGCN-master/data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("/home/zihe-leon/Desktop/RobustGCN-master/data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def perturb_features(features, ratio):
    features = features.toarray()
    pert_idx = np.random.choice(len(features), int(ratio*len(features)))
    np.save('perturbed_idx.npy', pert_idx)
    perturbed = []
    for row in range(len(features)):
        if row in pert_idx:
            arr = features[row]
            # flip_mask = [0]*int(0.8*len(arr)) + [1]*(len(arr)-int(0.8*len(arr)))
            # np.random.shuffle(flip_mask)
            # flip_mask = np.array(flip_mask, dtype=bool)
            p_idx = np.random.choice(len(arr), int(0.8*len(arr))) # 0.8
            arr[p_idx] = 1-arr[p_idx]
            # np.logical_not(arr, where=flip_mask, out=arr)    
            perturbed.append(arr)
        else:
            arr = features[row]
            perturbed.append(arr)
    perturbed_features = sp.csr_matrix(perturbed)
    return perturbed_features

# only used for pubmed (continuous feature)
def perturb_features_gaussian(features, ratio):
    features = features.toarray()
    pert_idx = np.random.choice(len(features), int(ratio*len(features)))
    np.save('perturbed_idx.npy', pert_idx)
    perturbed = []
    for row in range(len(features)):
        if row in pert_idx:
            arr = features[row]
            p_idx = np.random.choice(len(arr), int(0.8*len(arr))) # 0.8
            arr[p_idx] += np.random.normal(0, 0.5, 1)[0] 
            arr = np.clip(arr, 0, 1.2633097)
            perturbed.append(arr)
        else:
            arr = features[row]
            perturbed.append(arr)
    perturbed_features = sp.csr_matrix(perturbed)
    return perturbed_features


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    # print(features[0])
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)

def normalize_adj(adj, alpha):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, alpha).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj, alpha):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]), alpha)
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders, adj):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

def masked_softmax_cross_entropy(preds, labels, mask):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


def get_pert_rm_idx(thr_low, thr_high, variance_matrix):
    # variance_matrix has mean column
    # thr is the actual number like 0.22300164528433944
    rm_pert_idx = []
    for i in range(variance_matrix.shape[0]):
        # if variance_matrix['mean'][i] < thr_low:
        # if variance_matrix['mean'][i] > thr_high:
        if variance_matrix['mean'][i] < thr_low or variance_matrix['mean'][i] > thr_high:
            rm_pert_idx.append(i)
    return rm_pert_idx

def remove_pert(features, y_train, y_mask, adj, labels, y_val, val_mask, rm_pert_idx):
    features = features.toarray()
    adj = adj.toarray()
    cleaned_y = []
    cleaned_mask = []
    cleaned_features = []
    cleaned_labels = []
    cleaned_y_val = []
    cleaned_val_mask = []
    for i in range(len(features)):
        if not i in rm_pert_idx:
            cleaned_features.append(features[i])
            cleaned_y.append(y_train[i])
            cleaned_mask.append(y_mask[i])
            cleaned_labels.append(labels[i])
            cleaned_y_val.append(y_val[i])
            cleaned_val_mask.append(val_mask[i])
    cleaned_features = sp.csr_matrix(cleaned_features)
    cleaned_y = np.array(cleaned_y)
    cleaned_labels = np.array(cleaned_labels)
    cleaned_y_val = np.array(cleaned_y_val)
    # adj matrix
    adj = np.delete(adj, rm_pert_idx, 1) 
    cleaned_adj = np.delete(adj, rm_pert_idx, 0)
    cleaned_adj = sp.csr_matrix(cleaned_adj)
    return cleaned_features, cleaned_y, cleaned_mask, cleaned_adj, cleaned_labels, cleaned_y_val, cleaned_val_mask

def modify_pert(features, rm_pert_idx):
    features = features.toarray()
    good_features = np.delete(features, rm_pert_idx, 0)  
    mean_feature = np.mean(good_features, axis = 0)
    features[rm_pert_idx, :] = mean_feature
    features = sp.csr_matrix(features)
    return features


# version of perturbed_features used by gpc. only difference: also output labels (whether perturbed) for gpc
def perturb_features_gpc(features, ratio):
    features = features.toarray()
    y_train_gpc = [0]*features.shape[0]
    pert_idx = np.random.choice(len(features), int(ratio*len(features)))
    perturbed = []
    for row in range(len(features)):
        if row in pert_idx:
            y_train_gpc[row] = 1
            arr = features[row]
            p_idx = np.random.choice(len(arr), int(0.8*len(arr)))
            arr[p_idx] = 1-arr[p_idx]
            perturbed.append(arr)
        else:
            arr = features[row]
            perturbed.append(arr)
    # perturbed_features = sp.csr_matrix(perturbed)
    return perturbed, y_train_gpc

def get_gpc_train_data(perturbed_features, y_train_gpc, train_mask, test_mask, val_mask):
    # input all perturbed features, and all gpc label (whether perturbed)
    # return index, feature and labels used for gpc training
    gpc_idx = []
    gpc_feature = []
    gpc_y_train_gpc = []
    for idx in range(len(perturbed_features)):
        if not (train_mask[idx] or test_mask[idx] or val_mask[idx]):
            # if np.random.random()>0.8:
            gpc_idx.append(idx)
            gpc_feature.append(perturbed_features[idx])
            gpc_y_train_gpc.append(y_train_gpc[idx])

    return gpc_idx, gpc_feature, gpc_y_train_gpc