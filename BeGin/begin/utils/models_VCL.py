import torch
from torch import nn
from dgl.nn import GraphConv
import torch.nn.functional as F
from dgl.base import DGLError
import dgl.function as fn
from dgl.utils import expand_as_pair
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch_scatter import segment_csr
import math

from .models import AdaptiveLinear
        

class BCGNConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        norm="both",
        weight=True,
        bias=True,
        activation=None,
        allow_zero_in_degree=False,
        weight_prior_mu=None,
        weight_prior_logsd=None,
        bias_prior_mu=None,
        bias_prior_logsd=None,
        num_samples=10,
    ):
        super(BCGNConv, self).__init__()
        if norm not in ("none", "both", "right", "left"):
            raise DGLError(
                'Invalid norm value. Must be either "none", "both", "right" or "left".'
                ' But got "{}".'.format(norm)
            )
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.num_samples = num_samples

        if weight:
            self.weight_mu = nn.Parameter(torch.Tensor(in_feats, out_feats))
            self.weight_logsd = nn.Parameter(torch.Tensor(in_feats, out_feats))
            self.register_buffer(
                "weight_prior_mu",
                weight_prior_mu.detach().clone() if weight_prior_mu is not None
                else torch.zeros((in_feats, out_feats)),
            )
            self.register_buffer(
                "weight_prior_logsd",
                weight_prior_logsd.detach().clone() if weight_prior_logsd is not None
                else torch.zeros((in_feats, out_feats)),
            )
        else:
            self.register_parameter("weight_mu", None)
            self.register_parameter("weight_logsd", None)

        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(1, out_feats))
            self.bias_logsd = nn.Parameter(torch.Tensor(1, out_feats))
            self.register_buffer(
                "bias_prior_mu",
                bias_prior_mu.detach().clone() if bias_prior_mu is not None
                else torch.zeros((1, out_feats)),
            )
            self.register_buffer(
                "bias_prior_logsd",
                bias_prior_logsd.detach().clone() if bias_prior_logsd is not None
                else torch.zeros((1, out_feats)),
            )
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_logsd", None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        with torch.no_grad():
            if self.weight_mu is not None:
                nn.init.xavier_uniform_(self.weight_mu)
                self.weight_logsd.fill_(-6.0)
            if self.bias_mu is not None:
                nn.init.zeros_(self.bias_mu)
                self.bias_logsd.fill_(-6.0)

    def update_prior_and_reset(self):
        if self.weight_mu is not None:
            self.weight_prior_mu = self.weight_mu.detach().clone()
            self.weight_prior_logsd = self.weight_logsd.detach().clone()
        if self.bias_mu is not None:
            self.bias_prior_mu = self.bias_mu.detach().clone()
            self.bias_prior_logsd = self.bias_logsd.detach().clone()
        self.reset_parameters()

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value
    
    # def forward(self, graph, feat, weight=None, edge_weight=None):
    #     pred, kl = self._forward(graph, feat, weight, edge_weight)
    #     for _ in range(self.num_samples - 1):
    #         pred += self._forward(graph, feat, weight, edge_weight)[0]
    #     return pred / self.num_samples, kl

    def forward(self, graph, feat, weight=None, edge_weight=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )
            aggregate_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                aggregate_fn = fn.u_mul_e("h", "_edge_weight", "m")

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm in ["left", "both"]:
                degs = graph.out_degrees().to(feat_src).clamp(min=1)
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight_mu is not None:
                    raise DGLError(
                        "External weight is provided while at the same time the"
                        " module has defined its own weight parameter. Please"
                        " create the module with flag weight=False."
                    )
                kl = 0
            else:
                weight_q = Normal(
                    loc=self.weight_mu,
                    scale=torch.exp(self.weight_logsd),
                )
                weight_prior = Normal(
                    loc=self.weight_prior_mu,
                    scale=torch.exp(self.weight_prior_logsd),
                )
                kl = torch.sum(kl_divergence(weight_q, weight_prior))
                
                weight = weight_q.rsample()

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = torch.matmul(feat_src, weight)
                graph.srcdata["h"] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                rst = graph.dstdata["h"]
            else:
                # aggregate first then mult W
                graph.srcdata["h"] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                rst = graph.dstdata["h"]
                if weight is not None:
                    rst = torch.matmul(rst, weight)

            if self._norm in ["right", "both"]:
                degs = graph.in_degrees().to(feat_dst).clamp(min=1)
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            if self.bias_mu is not None:
                bias_q = Normal(
                    loc=self.bias_mu,
                    scale=torch.exp(self.bias_logsd),
                )
                bias_prior = Normal(
                    loc=self.bias_prior_mu,
                    scale=torch.exp(self.bias_prior_logsd),
                )
                kl += torch.sum(kl_divergence(bias_q, bias_prior))
                
                bias = bias_q.rsample()
                rst = rst + bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst, kl

    def extra_repr(self):
        summary = "in={_in_feats}, out={_out_feats}"
        summary += ", normalization={_norm}"
        if "_activation" in self.__dict__:
            summary += ", activation={_activation}"
        return summary.format(**self.__dict__)

class BGCNNode(nn.Module):
    def __init__(self, in_feats, n_classes, n_hidden, activation = F.relu, dropout=0.0, n_layers=3, incr_type='class', use_classifier=True, num_samples=10):
        super().__init__()
        self.num_samples = num_samples
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden
            self.convs.append(BCGNConv(in_hidden, out_hidden, "both", bias=False, allow_zero_in_degree=True))#, num_samples=num_samples))
            self.norms.append(nn.BatchNorm1d(out_hidden))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        if use_classifier:
            self.classifier = AdaptiveLinear(n_hidden, n_classes, bias=True, accum = False if incr_type == 'task' else True)
        else:
            self.classifier = None
    
    def update_prior_and_reset(self):
        for conv in self.convs:
            conv.update_prior_and_reset()
    
    def update_num_samples(self, num_samples):
        self.num_samples = num_samples
      
    def forward(self, graph, feat, task_masks=None):
        pred, kl = self._forward(graph, feat, task_masks=task_masks)
        for _ in range(self.num_samples-1):
            pred += self._forward(graph, feat, task_masks=task_masks)[0]
        return pred / self.num_samples, kl
            
    def _forward(self, graph, feat, task_masks=None):
        total_kl = 0
        h = feat
        h = self.dropout(h)
        for i in range(self.n_layers):
            conv, kl = self.convs[i](graph, h)
            total_kl += kl
            h = conv
            h = self.norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)
        if self.classifier is not None:
            h = self.classifier(h, task_masks)
        return h, total_kl
    
    # def forward_hat(self, graph, feat, hat_masks, task_masks=None):
    #     total_kl = 0
    #     h = feat
    #     h = self.dropout(h)
    #     for i in range(self.n_layers):
    #         conv, kl = self.convs[i](graph, h)
    #         total_kl += kl
    #         h = conv
    #         h = self.norms[i](h)
    #         h = self.activation(h)
    #         h = self.dropout(h)
            
    #         device = h.get_device()
    #         h=h*hat_masks[i].to(device).expand_as(h)
            
    #     if self.classifier is not None:
    #         h = self.classifier(h, task_masks)
    #     return h, total_kl
    
    # def forward_without_classifier(self, graph, feat, task_masks=None):
    #     total_kl = 0
    #     h = feat
    #     h = self.dropout(h)
    #     for i in range(self.n_layers):
    #         conv, kl = self.convs[i](graph, h)
    #         total_kl += kl
    #         h = conv
    #         h = self.norms[i](h)
    #         h = self.activation(h)
    #         h = self.dropout(h)
    #     return h, total_kl
    
    def bforward(self, blocks, feat, task_masks=None):
        pred, kl = self._bforward(blocks, feat, task_masks=task_masks)
        for _ in range(self.num_samples-1):
            pred += self._bforward(blocks, feat, task_masks=task_masks)[0]
        return pred / self.num_samples, kl
    
    def _bforward(self, blocks, feat, task_masks=None):
        total_kl = 0
        h = feat
        h = self.dropout(h) 
        for i in range(self.n_layers):
            conv, kl = self.convs[i](blocks[i], h)
            total_kl += kl
            h = conv
            h = self.norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)
        if self.classifier is not None:
            h = self.classifier(h, task_masks)
        return h, total_kl
    
    def observe_labels(self, new_labels, verbose=True):
        self.classifier.observe_outputs(new_labels, verbose=verbose)
    
    def get_observed_labels(self, tid=None):
        if tid is None or tid < 0:
            return self.classifier.observed
        else:
            return self.classifier.output_masks[tid]

class BLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
                
        self.weight_mu = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight_logsd = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.register_buffer("weight_prior_mu", torch.zeros((out_features, in_features)))
        self.register_buffer("weight_prior_logsd", torch.zeros((out_features, in_features)))
        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            self.bias_logsd = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            self.register_buffer("bias_prior_mu", torch.zeros((out_features)))
            self.register_buffer("bias_prior_logsd", torch.zeros((out_features)))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
            self.weight_logsd.fill_(-6.0)
            if self.bias_mu is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias_mu, -bound, bound)
                self.bias_logsd.fill_(-6.0)

    def update_prior_and_reset(self):
        self.weight_prior_mu = self.weight_mu.detach().clone()
        self.weight_prior_logsd = self.weight_logsd.detach().clone()
        if self.bias_mu is not None:
            self.bias_prior_mu = self.bias_mu.detach().clone()
            self.bias_prior_logsd = self.bias_logsd.detach().clone()
        self.reset_parameters()

    def forward(self, input):
        weight_q = Normal(
            loc=self.weight_mu,
            scale=torch.exp(self.weight_logsd),
        )
        weight_prior = Normal(
            loc=self.weight_prior_mu,
            scale=torch.exp(self.weight_prior_logsd),
        )
        kl = torch.sum(kl_divergence(weight_q, weight_prior))
        
        weight = weight_q.rsample()
        
        bias = 0
        if self.bias_mu is not None:
            bias_q = Normal(
                loc=self.bias_mu,
                scale=torch.exp(self.bias_logsd),
            )
            bias_prior = Normal(
                loc=self.bias_prior_mu,
                scale=torch.exp(self.bias_prior_logsd),
            )
            kl += torch.sum(kl_divergence(bias_q, bias_prior))
            
            bias = bias_q.rsample()
        
        return F.linear(input, weight, bias), kl

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

class BCGNGraph(nn.Module):
    def __init__(self, in_feats, n_classes, n_hidden, activation = F.relu, dropout=0.0, n_layers=4, n_mlp_layers=2, incr_type='class', readout='mean', node_encoder_fn=None, edge_encoder_fn=None, num_samples=10):
        super().__init__()
        self.num_samples = num_samples
        self.n_layers = n_layers
        self.n_mlp_layers = n_mlp_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        if node_encoder_fn is None:
            self.enc = BLinear(in_feats, n_hidden)
        else:
            self.enc = node_encoder_fn()
        for i in range(n_layers):
            in_hidden = n_hidden # if i > 0 else in_feats
            out_hidden = n_hidden
            self.convs.append(BCGNConv(in_hidden, out_hidden, "both", bias=False, allow_zero_in_degree=True))
            self.norms.append(nn.BatchNorm1d(out_hidden))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.mlp_layers = nn.ModuleList([BLinear(n_hidden // (1 << i), n_hidden // (1 << (i+1))) for i in range(n_mlp_layers)])
        self.classifier = AdaptiveLinear(n_hidden // (1 << n_mlp_layers), n_classes, bias=True, accum = False if incr_type == 'task' else True)
        self.readout_mode = readout
        
    def update_prior_and_reset(self):
        self.enc.update_prior_and_reset()
        for conv in self.convs:
            conv.update_prior_and_reset()
        for layer in self.mlp_layers:
            layer.update_prior_and_reset()
        
    def update_num_samples(self, num_samples):
        self.num_samples = num_samples
        
    def forward(self, graph, feat, task_masks=None, edge_weight=None, edge_attr=None, get_intermediate_outputs=False):
        pred, kl = self._forward(graph, feat, task_masks=task_masks, edge_weight=edge_weight, edge_attr=edge_attr, get_intermediate_outputs=get_intermediate_outputs)
        for _ in range(self.num_samples-1):
            pred += self._forward(graph, feat, task_masks=task_masks, edge_weight=edge_weight, edge_attr=edge_attr, get_intermediate_outputs=get_intermediate_outputs)[0]
        return pred / self.num_samples, kl
        
    def _forward(self, graph, feat, task_masks=None, edge_weight=None, edge_attr=None, get_intermediate_outputs=False):
        h, total_kl = self.enc(feat)
        h = self.dropout(h)
        inter_hs = []
        for i in range(self.n_layers):
            conv, kl = self.convs[i](graph, h, edge_weight=edge_weight)
            total_kl += kl
            h = conv
            h = self.norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)
            inter_hs.append(h)
            
        if self.readout_mode != 'none':
            # h0 = self.readout_fn(graph, h)
            # use deterministic algorithm instead
            ptrs = torch.cat((torch.LongTensor([0]).to(h.device), torch.cumsum(graph.batch_num_nodes(), dim=-1)), dim=-1)
            h1 = segment_csr(h, ptrs, reduce=self.readout_mode)
            # print((h1 - h0).abs().sum()) => 0
            h = h1
        for layer in self.mlp_layers:
            h, kl = layer(h)
            total_kl += kl
            h = self.activation(h)
            h = self.dropout(h)
        inter_hs.append(h)
        
        h = self.classifier(h, task_masks)
        if get_intermediate_outputs:
            return h, inter_hs
        else:
            return h, kl
        
    # def forward_hat(self, graph, feat, hat_masks, task_masks=None, edge_weight=None, edge_attr=None):
    #     h = self.enc(feat)
    #     h = self.dropout(h)
    #     device = h.get_device()
        
    #     h = h * hat_masks[0].to(device).expand_as(h)
    #     for i in range(self.n_layers):
    #         conv = self.convs[i](graph, h, edge_weight=edge_weight, edge_attr=edge_attr)
    #         h = conv
    #         h = self.norms[i](h)
    #         h = self.activation(h)
    #         h = self.dropout(h)
    #         h = h * hat_masks[1 + i].to(device).expand_as(h)
            
    #     if self.readout_mode != 'none':
    #         ptrs = torch.cat((torch.LongTensor([0]).to(h.device), torch.cumsum(graph.batch_num_nodes(), dim=-1)), dim=-1)
    #         h1 = segment_csr(h, ptrs, reduce=self.readout_mode)
    #         h = h1
            
    #     for layer in self.mlp_layers:
    #         h = layer(h)
    #         h = self.activation(h)
    #         h = self.dropout(h)
    #         h=h*hat_masks[1 + self.n_layers + i].to(device).expand_as(h)
            
    #     h = self.classifier(h, task_masks)
    #     return h
    
    # def forward_without_classifier(self, graph, feat, task_masks=None, edge_weight=None, edge_attr=None):
    #     h = self.enc(feat)
    #     h = self.dropout(h)
    #     for i in range(self.n_layers):
    #         conv = self.convs[i](graph, h, edge_weight=edge_weight, edge_attr=edge_attr)
    #         h = conv
    #         h = self.norms[i](h)
    #         h = self.activation(h)
    #         h = self.dropout(h)
    #     return h
    
    def observe_labels(self, new_labels, verbose=True):
        self.classifier.observe_outputs(new_labels, verbose=verbose)
    
    def get_observed_labels(self, tid=None):
        if tid is None or tid < 0:
            return self.classifier.observed
        else:
            return self.classifier.output_masks[tid]