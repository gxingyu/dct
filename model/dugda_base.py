import torch
from torch import nn
import torch.nn.functional as F

from pygda.nn.prop_gcn_conv import PropGCNConv
from pygda.nn.reverse_layer import GradReverse

from torch_geometric.nn import global_mean_pool


class DUGDABase(nn.Module):
    """
    Parameters
    ----------
    in_dim : int
        Input dimension of model.
    hid_dim : int
        Hidden dimension of model.
    num_classes : int
        Number of classes.
    num_layers : int, optional
        Total number of layers in model. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    mode : str, optional
        Mode for node or graph level tasks. Default: ``node``.
    **kwargs : optional
        Other parameters for the backbone.
    """

    def __init__(self,
                 in_dim,
                 hid_dim,
                 num_classes,
                 num_layers=1,
                 adv=False,
                 dropout=0.1,
                 act=F.relu,
                 mode='node',
                 **kwargs):
        super(DUGDABase, self).__init__()
        
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.adv = adv
        self.dropout = dropout
        self.act = act
        self.mode = mode

        self.convs = nn.ModuleList()

        self.convs.append(PropGCNConv(self.in_dim, self.hid_dim))

        for _ in range(self.num_layers - 1):
            self.convs.append(PropGCNConv(self.hid_dim, self.hid_dim))

        if self.mode == 'node':
            self.cls = PropGCNConv(self.hid_dim, self.num_classes)
        else:
            self.cls = nn.Linear(self.hid_dim, self.num_classes)

        if self.adv:
            self.domain_discriminator = nn.Linear(self.hid_dim, 2)
            
    def forward(self, data, prop_nums):
        if self.mode == 'node':
            x, edge_index, batch = data.x, data.edge_index, None
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.feat_bottleneck(x, edge_index, batch, prop_nums=prop_nums)
        x = self.feat_classifier(x, edge_index, batch, prop_nums=1)

        return x
    
    def feat_bottleneck(self, x, edge_index, batch, prop_nums=30):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, prop_nums)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.mode == 'graph':
            x = global_mean_pool(x, batch)

        return x
    
    def feat_classifier(self, x, edge_index, batch, prop_nums=1):
        if self.mode == 'node':
            x = self.cls(x, edge_index, prop_nums)
        else:
            x = self.cls(x)
        
        return x
    
    def domain_classifier(self, x, alpha):
        d_logit = self.domain_discriminator(GradReverse.apply(x, alpha))
        
        return d_logit
