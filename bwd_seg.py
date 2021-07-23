import paddle
import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F


paddle.set_device('cpu')


class MolTreeNode(object):
    def __init__(self, idx, nid, wid):
        self.idx = idx
        self.nid = nid
        self.wid = wid
        self.neighbors = []


class Tree(object):
    def __init__(self, nodes):
        self.nodes = nodes


def GRU(x, h_nei, W_z, W_r, U_r, W_h):
    hidden_size = x.shape[-1]
    sum_h = paddle.sum(h_nei, axis=1)
    z_input = paddle.concat([x, sum_h], axis=1)
    z = F.sigmoid(W_z(z_input))

    r_1 = paddle.reshape(W_r(x), shape=[-1, 1, hidden_size])
    r_2 = U_r(h_nei)
    r = F.sigmoid(r_1 + r_2)

    gated_h = r * h_nei
    sum_gated_h = paddle.sum(gated_h, axis=1)
    h_input = paddle.concat([x, sum_gated_h], axis=1)
    pre_h = F.tanh(W_h(h_input))
    new_h = (1.0 - z) * sum_h + z * pre_h
    return new_h


pred_loss = nn.CrossEntropyLoss(reduction='sum')
stop_loss = nn.BCEWithLogitsLoss(reduction='sum')
latent_size = 28
hidden_size = 450
MAX_NB = 15
embedding = nn.Embedding(531, hidden_size)
W_z = nn.Linear(2 * hidden_size, hidden_size)
U_r = nn.Linear(hidden_size, hidden_size, bias_attr=False)
W_r = nn.Linear(hidden_size, hidden_size)
W_h = nn.Linear(2 * hidden_size, hidden_size)
W = nn.Linear(hidden_size + latent_size, hidden_size)
# Stop Prediction Weights
U = nn.Linear(hidden_size + latent_size, hidden_size)
U_i = nn.Linear(2 * hidden_size, hidden_size)
# Output Weights
W_o = nn.Linear(hidden_size, 531)
U_o = nn.Linear(hidden_size, 1)


def dfs(stack, x, fa_idx):
    for y in x.neighbors:
        if y.idx == fa_idx: continue
        stack.append((x, y, 1))
        dfs(stack, y, x.idx)
        stack.append((y, x, 0))


def aggregate(hiddens, contexts, x_tree_vecs, mode):
    if mode == 'word':
        V, V_o = W, W_o
    elif mode == 'stop':
        V, V_o = U, U_o
    else:
        raise ValueError('aggregate mode is wrong')

    tree_contexts = paddle.index_select(axis=0, index=contexts, x=x_tree_vecs)
    input_vec = paddle.concat([hiddens, tree_contexts], axis=-1)
    output_vec = F.relu(V(input_vec))
    return V_o(output_vec)


n1 = MolTreeNode(0, 1, 133)
n2 = MolTreeNode(1, 2, 505)
n3 = MolTreeNode(2, 3, 133)
n1.neighbors.append(n2)
n2.neighbors.extend([n1, n3])
n3.neighbors.append(n2)

import copy

tree1 = Tree([n1, n2, n3])
tree2 = copy.deepcopy(tree1)
batches = [[tree1], [tree2]]

x_tree_vecs = paddle.randn([1, latent_size])
x_tree_vecs.stop_gradient = False


def forward(mol_batch, x_tree_vecs):
    global pred_loss, stop_loss, latent_size, hidden_size, MAX_NB, embedding, W_z, U_r, W_r, W_h, W, U, U_i, W_o, U_o
    pred_hiddens, pred_contexts, pred_targets = [], [], []

    traces = []
    for mol_tree in mol_batch:
        s = []
        dfs(s, mol_tree.nodes[0], -1)
        traces.append(s)
        for node in mol_tree.nodes:
            node.neighbors = []
    # Predict Root
    batch_size = len(mol_batch)

    pred_hiddens.append(paddle.zeros([len(mol_batch), hidden_size]))
    pred_targets.extend([mol_tree.nodes[0].wid for mol_tree in mol_batch])
    pred_contexts.append(paddle.to_tensor(list(range(batch_size))))

    max_iter = max([len(tr) for tr in traces])
    padding = paddle.zeros([hidden_size])
    padding.stop_gradient = True
    h = {}
    # print('max_iter', max_iter)

    for t in range(max_iter):
        prop_list = []
        batch_list = []
        for i, plist in enumerate(traces):
            if t < len(plist):
                prop_list.append(plist[t])
                batch_list.append(i)

        cur_x = []
        cur_h_nei, cur_o_nei = [], []

        for node_x, real_y, _ in prop_list:
            # Neighbors for message passing (target not included)
            cur_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors if node_y.idx != real_y.idx]
            pad_len = MAX_NB - len(cur_nei)
            cur_h_nei.extend(cur_nei)
            cur_h_nei.extend([padding] * pad_len)
            # Neighbors for stop prediction (all neighbors)
            cur_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]
            pad_len = MAX_NB - len(cur_nei)

            # Current clique embedding
            cur_x.append(node_x.wid)
        # Clique embedding
        cur_x = paddle.to_tensor(cur_x)
        cur_x = embedding(cur_x)
        # Message passing
        cur_h_nei = paddle.reshape(paddle.stack(cur_h_nei, axis=0), shape=[-1, MAX_NB, hidden_size])
        new_h = GRU(cur_x, cur_h_nei, W_z, W_r, U_r, W_h)
        # Gather targets
        pred_list = []
        # stop_target = []
        for i, m in enumerate(prop_list):
            node_x, node_y, direction = m
            x, y = node_x.idx, node_y.idx
            h[(x, y)] = new_h[i]
            node_y.neighbors.append(node_x)
            if direction == 1:
                pred_list.append(i)

        # Hidden states for stop prediction
        cur_batch = paddle.to_tensor((batch_list))
        # Hidden states for clique prediction
        if len(pred_list) > 0:
            batch_list = [batch_list[i] for i in pred_list]
            cur_batch = paddle.to_tensor(batch_list)
            pred_contexts.append(cur_batch)

            cur_pred = paddle.to_tensor(pred_list)
            pred_hiddens.append(paddle.index_select(axis=0, index=cur_pred, x=new_h))
            # pred_targets.extend(pred_target)
    # Predict next clique
    pred_contexts = paddle.concat(pred_contexts, axis=0)
    pred_hiddens = paddle.concat(pred_hiddens, axis=0)
    pred_scores = aggregate(pred_hiddens, pred_contexts, x_tree_vecs, 'word')
    tmp = paddle.sum(pred_scores)
    return tmp


for i, batch in enumerate(batches):
    loss = forward(batch, x_tree_vecs)
    print('i:%s, loss:%s' % (i, loss))
    loss.backward()