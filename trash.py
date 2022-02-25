import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence

import numpy as np

class GraphConvolution(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        dropout=0.2,
        bias=True,
        activation=F.relu,
    ):
        super().__init__()

        if dropout:
            self.dropout = dropout
        else:
            self.dropout = 0.0

        self.bias = bias
        self.activation = activation

        def glorot(shape, name=None):
            """Glorot & Bengio (AISTATS 2010) init."""
            init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
            init = torch.FloatTensor(shape[0], shape[1]).uniform_(
                -init_range, init_range
            )
            return init

        self.weight = nn.Parameter(glorot((input_dim, output_dim)))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, inputs):
        # node feature, adj matrix
        # D^(-1/2).A.D^(-1/2).H_i.W_i
        # with H_0 = X (init node features)
        # V, A
        x, support = inputs

        x = F.dropout(x, self.dropout)
        xw = torch.mm(x, self.weight)
        out = torch.sparse.mm(support, xw)

        if self.bias is not None:
            out += self.bias

        if self.activation is None:
            return out, support
        else:
            return self.activation(out), support

class LinearEmbedding(torch.nn.Module):

    def __init__(self, input_size, output_size, use_act="relu"):
        super().__init__()
        self.C = output_size
        self.F = input_size

        self.W = nn.Parameter(torch.FloatTensor(self.F, self.C))
        self.B = nn.Parameter(torch.FloatTensor(self.C))

        if use_act == "relu":
            self.act = torch.nn.ReLU()
        elif use_act == "softmax":
            self.act = torch.nn.Softmax(dim=-1)
        else:
            self.act = None

        nn.init.xavier_normal_(self.W)
        nn.init.normal_(self.B, mean=1e-4, std=1e-5)

    def forward(self, V):
        # V shape B,N,F
        # V: node features
        V_out = torch.matmul(V, self.W) + self.B
        if self.act:
            V_out = self.act(V_out)

        return V_out
        
class GCN(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128, 64],
                 bias=True, dropout_rate=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.bias = bias
        self.dropout_rate = dropout_rate

        gcn_layers = []
        for index, (h1, h2) in enumerate(
            zip(self.hidden_dims[:-1], self.hidden_dims[1:])):
            gcn_layers.append(
                GraphConvolution(
                    h1,
                    h2,
                    activation=None if index == len(self.hidden_dims) else F.relu,
                    bias=self.bias,
                    dropout=self.dropout_rate,
                    is_sparse_inputs=False
                )
            )

        self.layers = nn.Sequential(*gcn_layers)
        self.linear1 = LinearEmbedding(input_dim, self.hidden_dims[0], use_act='relu')
        self.linear2 = LinearEmbedding(self.hidden_dims[-1], self.output_dim, use_act='relu')

    def forward(self, inputs):
        # features, adj
        x, support = inputs
        x = self.linear1(x)
        x = F.dropout(x, p=self.dropout_rate)
        x, _ = self.layers((x, support))
        x = self.linear2(x)
        return x, support

import scipy.sparse as sp
import os


def weight_mask(labels):
    label_classes = copy.deepcopy(LABELS)
    weight_dict = {}
    for k in label_classes:
        if k == "other" or k == 'invoice':
            weight_dict[k] = 0.8
        else:
            weight_dict[k] = 1.0
    tmp_list = []
    for arr in labels:
        index = np.argmax(arr)
        tmp_list.append(weight_dict[label_classes[index]])
    return np.array(tmp_list)

def weighted_loss(preds, labels, weight=None, class_weight=False, device='cuda'):
    """Softmax cross-entropy loss with weights."""
    if class_weight:
        if weight is not None:
            weight = torch.tensor(weight).float().to(device)
        loss = F.cross_entropy(preds, labels, reduction='none', weight=weight)
    else:
        # sample weight
        # https://discuss.pytorch.org/t/how-to-weight-the-loss/66372/3
        # https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264
        loss = F.cross_entropy(preds, labels, reduction='none')
        if weight is not None:
            weight = weight.float()
            loss *= weight
    loss = loss.mean()
    return loss

def load_single_graph(img_id):
    adj = scipy.sparse.load_npz(os.path.join(matrix_dir, img_id + "_adj.npz"))
    features = np.load(os.path.join(matrix_dir, img_id + "_feature.npy"), allow_pickle=True)
    labels = np.load(os.path.join(matrix_dir, img_id + "_label.npy"), allow_pickle=True)
    weights_mask = weight_mask(labels)
    return adj, features, labels, weights_mask

def cal_accuracy(out, label):
    "Accuracy in single graph."
    pred = out.argmax(dim=1)
    correct = torch.eq(pred, label).float()
    acc = correct.mean()
    return acc

def convert_sparse_input(adj, features):
    supports = preprocess_adj(adj)
    # coords, values in coord
    m = torch.from_numpy(supports[0]).long()
    n = torch.from_numpy(supports[1])
    support = torch.sparse.FloatTensor(m.t(), n, supports[2]).float()

    features = [
        torch.tensor(idxs, dtype=torch.float32).to(device)
        if torch.cuda.is_available()
        else torch.tensor(idxs, dtype=torch.float32)
        for idxs in features
    ]
    features = torch.stack(features).to(device)

    if torch.cuda.is_available():
        m = m.to(device)
        n = n.to(device)
        support = support.to(device)
    return features, support

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))  # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # D^-0.5
    return (
        adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    )  # D^-0.5AD^0.5

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

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + scipy.sparse.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def convert_loss_input(y_train, weight_mask):
    train_label = torch.from_numpy(y_train).long()
    weight_mask = torch.from_numpy(weight_mask)

    if torch.cuda.is_available():
        train_label = train_label.to(device)
        weight_mask = weight_mask.to(device)
    train_label = train_label.argmax(dim=1)

    return train_label, weight_mask

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))  # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # D^-0.5
    return (
        adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    )  # D^-0.5AD^-0.5

# training model
import random
import time


for epoch in range(20):
    net.train()
    random.shuffle(train_ids)
    t1 = time.time()
    train_losses = 0
    train_accs = []

    batch_losses = []
    # simple training with batch_size = 1
    for img_index, img_id in tqdm.notebook.tqdm(enumerate(train_ids)):
        adj, features, train_labels, weight_mask_ = load_single_graph(img_id)
        features, support = convert_sparse_input(adj, features)
        train_labels, weight_mask_ = convert_loss_input(train_labels, weight_mask_)
        support = support.to(device)
        out = net((features, support))[0]
        loss = weighted_loss(out, train_labels, _class_weights, class_weight=True)

        train_losses += loss.item()
        batch_losses.append(loss.item())
        if img_index % 100 == 0:
            print("\ttrain loss: {:.5f} ".format(np.mean(batch_losses)))
            batch_losses = []

        acc = cal_accuracy(out, train_labels)
        train_accs.append(acc.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_losses /= (img_index + 1)
    acc = np.mean(train_accs)
    t2 = time.time()
    print(
        "Epoch:",
        "%04d" % (epoch + 1),
        "time: {:.5f}, loss: {:.5f}, acc: {:.5f}".format(
            (t2 - t1), train_losses, acc.item()
        ),
    )