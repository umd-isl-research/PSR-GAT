from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType, OptTensor)
import torch
from torch import Tensor
from torch.nn import Parameter, Linear
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, add_remaining_self_loops, softmax

from util import device

def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes

# consider edge weight when add or remove self loop
def add_self_loops_partial(edge_index, edge_weight=None, fill_value=1.0, num_nodes=None):
    # for existing self loop, keep them, for nodes do not have self loop, add self loop with specified edge weight
    assert edge_weight is not None, "this function is not applicable for edge_weight = None"
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index
    mask = row == col
    masked_weight = edge_weight[mask]
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    loop_weight = torch.full((num_nodes,), fill_value).to(device)
    loop_weight[row[mask]] = masked_weight
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    edge_weight = torch.cat([edge_weight, loop_weight], dim=0)
    assert edge_index.shape[-1] == edge_weight.shape[0]
    return edge_index, edge_weight


class ConditionalGATConv(MessagePassing):  # compatible with PyG=1.6.3
    _alpha: OptTensor
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 heads: int = 1,
                 concat: bool = True,
                 negative_slope: float = 0.2,
                 dropout: float = 0.,
                 add_self_loops: bool = True,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(ConditionalGATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2*out_channels))  # this is in FGNN

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor],
                edge_index: Adj,
                size: Size = None,
                return_attention_weights=None,
                edge_attr=None):
        """
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        """
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)

            alpha_l = (x_l * self.att_l).sum(dim=-1)  # not needed in FGNN
            alpha_r = (x_r * self.att_r).sum(dim=-1)  # not needed in FGNN
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                if edge_attr is None:
                    edge_index, _ = remove_self_loops(edge_index)
                    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
                else:
                    edge_index, edge_attr = add_self_loops_partial(edge_index, edge_attr, num_nodes=num_nodes)  # keep existing self loop and add self loop for the other nodes
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=(alpha_l, alpha_r), size=size, edge_attr=edge_attr)  # alpha=(alpha_l, alpha_r),

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, x_i: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int],
                edge_attr: Tensor) -> Tensor:
        # i do not know what it means,
        alpha_v163 = alpha_j if alpha_i is None else alpha_j + alpha_i  # this is v 1.6.2

        # alpha = (torch.cat([x_i, x_j, edge_attr.view(-1, 1).repeat(1, self.heads).view(-1, self.heads, 1)], dim=-1) * self.att).sum(dim=-1)  # this is in FGNN
        alpha = (torch.cat([x_i, x_j], dim=-1) * edge_attr.view(-1, 1).repeat(1, self.heads).view(-1, self.heads, 1) * self.att).sum(dim=-1)  # this is my design
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class ConditionalGATConv_v142(MessagePassing):  # compatible with PyG=1.4.2
    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0,
                 bias=True, weighted=True):
        super(ConditionalGATConv_v142, self).__init__('add')
        self.weighted = weighted
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))
        # self.att = Parameter(torch.Tensor(1, heads, out_channels))  # if used for elementary product
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))  # (1, heads, 2 * out_channels)  (heads*out_channels*2, heads)
        # self.cond_matrix = RandomLayer()

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr=None, size=None, return_attention_weights=False):
        edge_index, edge_attr = add_self_loops_partial(edge_index, edge_attr)
        x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)
        out = self.propagate(edge_index, x=x, size=size, edge_attr=edge_attr, return_attention_weights=return_attention_weights)
        if return_attention_weights:
            alpha, self.alpha = self.alpha, None
            return out, alpha
        else:
            return out

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr, return_attention_weights):
        # Compute attention coefficients.
        if edge_attr is not None:
            # alpha = ((torch.cat([x_i, x_j], dim=-1) * self.att) * edge_attr.view(-1, 1, 1)).sum(dim=-1)
            # alpha = (torch.cat([x_i, x_j, edge_attr.view(-1, 1).repeat(1, x_i.shape[1]).view(-1, x_i.shape[1], 1)], dim=-1) * self.att).sum(dim=-1)# this is what in FGNN
            # alpha = (x_i* edge_attr.view(-1, 1).repeat(1, x_i.shape[1]).view(-1, x_i.shape[1], 1) * self.att).sum(dim=-1)  # tried
            alpha = (torch.cat([x_i, x_j], dim=-1) * edge_attr.view(-1, 1).repeat(1, x_i.shape[1]).view(-1, x_i.shape[1], 1) * self.att).sum(dim=-1)
            # alpha = torch.mm(x_i.view(-1, self.heads*self.out_channels)* edge_attr.view(-1, 1), self.att) # use matrix multiply
        else:
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)
        if return_attention_weights:
            self.alpha = alpha

        # Sample attention coefficients stochastically.
        #alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class Set2Set_GRU(torch.nn.Module):  #  this is written according to my understanding, same to GRUSet2Set
    def __init__(self, in_channels, processing_steps, num_layers=1):
        super(Set2Set_GRU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.num_layers = num_layers
        self.gru = torch.nn.GRU(self.out_channels, self.in_channels, num_layers)
        self.reset_parameters()

    def reset_parameters(self):
        self.gru.reset_parameters()

    def forward(self, x, batch):
        """"""
        batch_size = batch.max().item() + 1
        h = x.new_zeros((self.num_layers, batch_size, self.in_channels))
        q_star = x.new_zeros(batch_size, self.out_channels)

        for i in range(self.processing_steps):
            q, h = self.gru(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.in_channels)
            e = (x * q[batch]).sum(dim=-1, keepdim=True)
            a = softmax(e, batch, num_nodes=batch_size)
            r = scatter_add(a * x, batch, dim=0, dim_size=batch_size)
            q_star = torch.cat([q, r], dim=-1)
        return q_star
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)