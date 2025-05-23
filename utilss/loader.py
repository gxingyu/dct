import torch
import random
import numpy as np

from models.ScoreNetwork_A import ScoreNetworkA
from models.ScoreNetwork_X import ScoreNetworkX, ScoreNetworkX_GMH
from utilss.sde import VPSDE, VESDE, subVPSDE

from utilss.losses import get_sde_loss_fn
from utilss.solver import get_pc_sampler, S4_solver
from utilss.ema import ExponentialMovingAverage

def load_device():
    if torch.cuda.is_available():
        device = list(range(torch.cuda.device_count()))
    else:
        device = 'cpu'
    return device


def load_model(params):
    params_ = params.copy()
    model_type = params_.pop('model_type', None)
    if model_type == 'ScoreNetworkX':
        model = ScoreNetworkX(**params_)
    elif model_type == 'ScoreNetworkA':
        model = ScoreNetworkA(**params_)
    elif model_type == 'ScoreNetworkX_GMH':
        model = ScoreNetworkX_GMH(**params_)
    else:
        raise ValueError(f"Model Name <{model_type}> is Unknown")
    return model


def load_model_optimizer(params, config_train, device):
    model = load_model(params)
    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f'cuda:{device[0]}')

    optimizer = torch.optim.Adam(model.parameters(), lr=config_train.lr,
                                 weight_decay=config_train.weight_decay)
    scheduler = None
    if config_train.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config_train.lr_decay)

    return model, optimizer, scheduler


def load_ema(model, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    return ema


def load_ema_from_ckpt(model, ema_state_dict, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    ema.load_state_dict(ema_state_dict)
    return ema


def load_data(config, get_graph_list=False):
    if config.data.data in ['QM9', 'ZINC250k']:

        from utilss.data_loader_mol import dataloader
        return dataloader(config, get_graph_list)
    else:

        from utilss.data_loader import dataloader
        return dataloader(config, get_graph_list)


def load_batch(batch, device):
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    x_b = batch[0].to(device_id)
    adj_b = batch[1].to(device_id)
    return x_b, adj_b


def load_sde(config_sde):
    sde_type = config_sde.type
    beta_min = config_sde.beta_min
    beta_max = config_sde.beta_max
    num_scales = config_sde.num_scales

    if sde_type == 'VP':
        sde = VPSDE(beta_min=beta_min, beta_max=beta_max, N=num_scales)
    elif sde_type == 'VE':
        sde = VESDE(sigma_min=beta_min, sigma_max=beta_max, N=num_scales)
    elif sde_type == 'subVP':
        sde = subVPSDE(beta_min=beta_min, beta_max=beta_max, N=num_scales)
    else:
        raise NotImplementedError(f"SDE class {sde_type} not yet supported.")
    return sde


def load_loss_fn(config):
    reduce_mean = config.train.reduce_mean
    sde_x = load_sde(config.sde.x)
    sde_adj = load_sde(config.sde.adj)

    loss_fn = get_sde_loss_fn(sde_x, sde_adj, train=True, reduce_mean=reduce_mean, continuous=True,
                              likelihood_weighting=False, eps=config.train.eps)
    return loss_fn


def load_sampling_fn(config_train, config_module, config_sample, device, subgraph_node_nums):
    sde_x = load_sde(config_train.sde.x)
    sde_adj = load_sde(config_train.sde.adj)
    max_node_num = config_train.data.max_node_num

    if config_module.predictor == 'S4':
        get_sampler = S4_solver
    else:
        get_sampler = get_pc_sampler
    node_num = subgraph_node_nums
    shape_x = (config_train.data.batch_size, node_num, config_train.data.max_sample_feat_num)
    shape_adj = (config_train.data.batch_size, node_num, node_num)

    sampling_fn = get_sampler(sde_x=sde_x, sde_adj=sde_adj, shape_x=shape_x, shape_adj=shape_adj,
                              predictor=config_module.predictor, corrector=config_module.corrector,
                              snr=config_module.snr, scale_eps=config_module.scale_eps,
                              n_steps=config_module.n_steps,
                              probability_flow=config_sample.probability_flow,
                              continuous=True, denoise=config_sample.noise_removal,
                              eps=config_sample.eps, device=device)
    return sampling_fn


def load_model_params(config):
    config_m = config.model
    max_feat_num = config.data.max_feat_num
    max_sample_feat_num = config.data.max_sample_feat_num
    if 'GMH' in config_m.x:
        params_x = {'model_type': config_m.x, 'max_feat_num': max_feat_num, 'depth': config_m.depth,
                    'nhid': config_m.nhid, 'num_linears': config_m.num_linears,
                    'c_init': config_m.c_init, 'c_hid': config_m.c_hid, 'c_final': config_m.c_final,
                    'adim': config_m.adim, 'num_heads': config_m.num_heads, 'conv': config_m.conv}
    else:
        params_x = {'model_type': config_m.x, 'max_feat_num': max_sample_feat_num, 'depth': config_m.depth,
                    'nhid': config_m.nhid}
    params_adj = {'model_type': config_m.adj, 'max_feat_num': max_feat_num, 'max_node_num': config.data.max_node_num,
                  'nhid': config_m.nhid, 'num_layers': config_m.num_layers, 'num_linears': config_m.num_linears,
                  'c_init': config_m.c_init, 'c_hid': config_m.c_hid, 'c_final': config_m.c_final,
                  'adim': config_m.adim, 'num_heads': config_m.num_heads, 'conv': config_m.conv}
    return params_x, params_adj

