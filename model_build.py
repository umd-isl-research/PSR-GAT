import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from custermized_gcn_layers import Set2Set_GRU, ConditionalGATConv
from util import duration2weight, norm_torchtensor

def save_model(net, filename):
    """Save trained model."""
    torch.save(net.state_dict(), filename)
    print("save model to: {}".format(filename))

def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        nn.init.kaiming_normal_(layer.weight) # layer.weight.data.normal_(0.0, 0.02)
        layer.bias.data.fill_(0.01)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0.01)  # original 0
    elif layer_name.find("Linear") != -1:
        nn.init.xavier_normal_(layer.weight)
        layer.bias.data.fill_(0.01)


def zero_weights(layer):
    if type(layer) != nn.LeakyReLU: # and type(layer)!= TopKPooling:
        nn.init.constant_(layer.weight, 0.0)


def get_batches(data_set, batch_size=64, shuffer=False):
    if shuffer:
        ind = np.random.permutation(len(data_set))
        dataset = data_set[ind]  #shuffle
    #n_batches = len(u)//config.batch_size
    # u, i, l = u[:n_batches*config.batch_size], i[:n_batches*config.batch_size], l[::n_batches*config.batch_size]
    for b in range(0,data_set.shape[0], batch_size):
        yield data_set[b:b+batch_size]


embed_dim = 64
class PSR_GAT(torch.nn.Module): 
    def __init__(self, num_items=239, head=1, use_multista=True): #239 in batch1, 231 in batch2, without counting default screens
        super(PSR_GAT, self).__init__()
        self.num_items = num_items
        self.use_multista = use_multista
        self.conv1 = ConditionalGATConv(embed_dim, 64, heads=head) # PrefGATConv  ConditionalGATConv
        self.pool1 = TopKPooling(64*head, ratio=0.9)
        self.conv2 = ConditionalGATConv(64*head, 64, heads=head, concat=False) # concat=False, means use mean instead of concatenation for output of GAT
        self.pool2 = TopKPooling(64, ratio=0.9)
        self.item_embedding = torch.nn.Embedding(num_embeddings=num_items, embedding_dim=embed_dim)
        self.lin1 = torch.nn.Linear(64, 64)
        self.lin2 = torch.nn.Linear(64, 64)
        self.act1 = torch.nn.LeakyReLU()
        self.act2 = torch.nn.LeakyReLU()

        self.gp = Set2Set_GRU(in_channels=64, processing_steps=5)  # my realization using GRU
        self.lin3 = torch.nn.Linear(64*2+1, 64)
        if use_multista:
            self.sta_weight = nn.Parameter(torch.Tensor(6))
            nn.init.uniform_(self.sta_weight, 0.01, 1.0)
    def forward(self, data, sta=None):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, duration2weight(data.edge_attr)
        last2_edge_attr = duration2weight(data.last2_edge_attr)
        # edge_index, edge_attr = add_self_loops_partial(edge_index, edge_attr)
        scr_freq_m, scr_dur_m = norm_torchtensor(data.scr_freq_m), norm_torchtensor(data.scr_dur_m)
        scr_freq_w, scr_dur_w = norm_torchtensor(data.scr_freq_w), norm_torchtensor(data.scr_dur_w)
        scr_freq_d, scr_dur_d = norm_torchtensor(data.scr_freq_d), norm_torchtensor(data.scr_dur_d)

        emb_item = self.item_embedding(x).squeeze(1)
        x, (edge_index, alpha) = self.conv1(emb_item, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x = F.leaky_relu(x)
        assert edge_index.size(1) == alpha.size(0), "# of edges and attributes do not match"
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr=alpha.mean(dim=-1), batch=batch)
        x, (edge_index, alpha) = self.conv2(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x = F.leaky_relu(x)
        assert edge_index.size(1) == alpha.size(0), "# of edges and attributes do not match"
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr=alpha.mean(dim=-1), batch=batch)

        x = self.act1(self.lin1(x))
        x = self.act2(F.dropout(self.lin2(x), p=0.5, training=self.training))
        x = self.gp(x, batch)
        x = torch.cat([x, last2_edge_attr[:,0][:,None]], dim=1)
        x = self.lin3(x)
        if sta is None:  # we tried different settings
            scr_freq_m, scr_dur_m = torch.exp(scr_freq_m), torch.exp(scr_dur_m)
            scr_freq_w, scr_dur_w = torch.exp(scr_freq_w), torch.exp(scr_dur_w)
            scr_freq_d, scr_dur_d = torch.exp(scr_freq_d), torch.exp(scr_dur_d)
            pref_fuse = (self.sta_weight[0] * scr_freq_m + self.sta_weight[1] * scr_dur_m + self.sta_weight[2] * scr_freq_w + \
                self.sta_weight[3] * scr_dur_w + self.sta_weight[4] * scr_freq_d + self.sta_weight[5] * scr_dur_d)# /self.sta_weight.sum()
            return torch.mul(torch.mm(x, self.item_embedding.weight.t()), pref_fuse)
        else:
            if self.use_multista:
                scr_freq_m, scr_dur_m = torch.exp(scr_freq_m), torch.exp(scr_dur_m)
                scr_freq_w, scr_dur_w = torch.exp(scr_freq_w), torch.exp(scr_dur_w)
                scr_freq_d, scr_dur_d = torch.exp(scr_freq_d), torch.exp(scr_dur_d)
                pref_fuse = (self.sta_weight[0] * scr_freq_m + self.sta_weight[1] * scr_dur_m + self.sta_weight[2] * scr_freq_w + \
                    self.sta_weight[3] * scr_dur_w + self.sta_weight[4] * scr_freq_d + self.sta_weight[5] * scr_dur_d)  # /self.sta_weight.sum()
                return torch.mul(torch.mul(torch.mm(x, self.item_embedding.weight.t()), pref_fuse), sta.unsqueeze(0))
            else:
                return torch.mul(torch.mm(x, self.item_embedding.weight.t()), sta.unsqueeze(0))


if __name__ == "__main__":
    pass
