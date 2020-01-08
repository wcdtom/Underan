from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim

from gae.model import GCNModelVAE
from gae.optimizer import loss_function
from gae.utils import *
# load_corpus, load_data, mask_test_edges, preprocess_graph, get_roc_score, sparse_to_tuple

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=16, help='Number of units in hidden layer 1.')
# parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='20ng', help='type of dataset.')

args = parser.parse_args()


def gae_for(args):
    print("Using {} dataset".format(args.dataset_str))
    adj, features = load_corpus(args.dataset_str)
    # n_nodes, feat_dim = features.shape
    # print(n_nodes, feat_dim)
    print(type(features))

    print(adj)
    # print(adj[0], adj[1])
    features = sp.identity(features.shape[0])  # featureless

    # print(adj.shape)
    # print(features.shape)

    # Some preprocessing
    features = preprocess_features(features)


    adj_norm = preprocess_adj(adj)
    num_supports = 1
    # model_func = GCN
    adj_norm = torch.FloatTensor(adj_norm.toarray())
    features = torch.FloatTensor(features.toarray())
    n_nodes, feat_dim = features.shape
    print(n_nodes, feat_dim)
    print(type(features))
    print(type(adj_norm))
    print(features.shape)
    print(adj_norm.shape)
    # n_nodes, feat_dim = features.shape
    # print(n_nodes, feat_dim)
    # Store original adjacency matrix (without diagonal entries) for later
    # adj_orig = adj
    # adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    # adj_orig.eliminate_zeros()

    # modified/added by hollis
    # adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    # Remove diagonal elements

    # adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    # adj.eliminate_zeros()
    # Check that diag is zero:
    # assert np.diag(adj.todense()).sum() == 0

    # adj_train = sp.csr_matrix(adj)
    # adj_train = adj_train + adj_train.T

    # Some preprocessing
    # adj_norm = normalize_adj(adj)

    # adj_label = adj_train + sp.eye(adj_train.shape[0])

    # adj_label = sparse_to_tuple(adj_label)

    # adj_label = torch.FloatTensor(adj_label.toarray())
    # adj_label = np.array(adj_label, dtype=float)
    # adj_label = torch.FloatTensor(adj_label)

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()

    # added by hollis
    # pos_weight = torch.from_numpy(np.array(pos_weight))

    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    # model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    model = GCNModelVAE(feat_dim, args.hidden1, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    hidden_emb = None
    for epoch in range(args.epochs):
        print("in epoch")
        t = time.time()
        model.train()
        optimizer.zero_grad()
        # recovered, mu, logvar = model(features, adj_norm)

        print("before model")
        recovered, mu, logvar = model(features, adj_norm)

        print("before loss")
        loss = loss_function(preds=recovered, labels=adj_norm,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm)
        #loss = loss_function(preds=recovered, labels=adj_label,
        #                     mu=mu, logvar=logvar, n_nodes=n_nodes,
        #                     norm=norm, pos_weight=pos_weight)
        print("befor backword")
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        hidden_emb = mu.data.numpy()
        hidden_emb = np.array(hidden_emb)

        if epoch == 1:
            fni = "./result/emb_init.txt"
            hidden_emb = np.array(hidden_emb)
            np.savetxt(fni, hidden_emb)

        if epoch == args.epochs-1:
            fnf = "./result/emb.txt"
            hidden_emb = np.array(hidden_emb)
            np.savetxt(fnf, hidden_emb)

        #roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
              # "val_ap=", "{:.5f}".format(ap_curr),
              "time=", "{:.5f}".format(time.time() - t)
              )

    print("Optimization Finished!")

    # roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    # print('Test ROC score: ' + str(roc_score))
    # print('Test AP score: ' + str(ap_score))


if __name__ == '__main__':
    gae_for(args)
