import torch
import torch.nn as nn
from utils import get_path_norm
import math

eps = 1e-6


def collect_other_norm(model, lambd, regularize_bias=False):
    total_l2_norm = 0.
    for layer in model.other_layers:
        this_l2_norm = torch.pow(layer.weight, 2).sum()
        total_l2_norm += this_l2_norm
        if regularize_bias:
            this_l2_norm = torch.pow(layer.bias, 2).sum()
            total_l2_norm += this_l2_norm
    if regularize_bias:
        for grouped_layer in model.grouped_layers:
            v_b_l2_norm = torch.pow(grouped_layer[1].bias, 2).sum()
            total_l2_norm += v_b_l2_norm
    return 0.5 * lambd * total_l2_norm


def collect_grouped_norm(model, lambd, loss_term, regularize_bias=False):
    total = 0.
    w_norm_deg = 2
    v_norm_deg = 2
    if loss_term == 'wd':
        for grouped_layer in model.grouped_layers:
            w = grouped_layer[0]
            w_l2_norm = torch.pow(w.weight, 2).sum()
            if regularize_bias:
                w_l2_norm += torch.pow(w.bias, 2).sum()
            v = grouped_layer[1]
            v_l2_norm = torch.pow(v.weight, 2).sum()
            if regularize_bias:
                v_l2_norm += torch.pow(v.bias, 2).sum()
            total += 0.5 * lambd * (w_l2_norm + v_l2_norm)
    elif loss_term == 'pn':
        for grouped_layer in model.grouped_layers:
            pn = get_path_norm(grouped_layer, w_norm_deg, v_norm_deg, requires_grad=True, include_bias=regularize_bias)
            total += pn.sum() * lambd
    else:
        raise NotImplementedError("Only support loss term wd | pn")
    return total


def prox_grad_upd_v(v, lam):
    """
    :param v: shape [out_dim, N]
    """
    v_norm = torch.linalg.vector_norm(v, dim=0, ord=2)  # [N,]
    v_upd = torch.where(v_norm <= lam, torch.zeros_like(v),
                        v - lam * v / torch.clip(v_norm[None, :], min=eps))  # [out_dim, N]
    return v_upd


def prox_grad_upd_v_conv(v, lam):
    """
    :param v: shape [out_dim, N, kernel, kernel]
    """
    assert len(v.shape) == 4, "Weight dimension has to be 4 for convolutional prox." \
                              " Got %d dimensions for v_k" % (len(v.shape))
    v_norm = torch.linalg.vector_norm(v, dim=(0, 2, 3), ord=2)  # [N,]
    v_upd = torch.where((v_norm <= lam)[:, None, None], torch.zeros_like(v),
                        v - lam * v / torch.clip(v_norm[:, None, None], min=eps))  # [out_dim, N, K, K]

    return v_upd


def prox_grad_upd_w(w, w_b, regularize_bias=False):
    """
    :param w: shape [N, in_dim]
    """
    if regularize_bias:
        tmp_w = torch.cat([w, w_b[:, None]], dim=1)  # [N, input_dim + 1]
    else:
        tmp_w = w
    w_norm = torch.linalg.vector_norm(tmp_w, dim=1, ord=2)  # [N,]
    w_norm = torch.clip(w_norm, min=eps)
    w_upd = w / w_norm[:, None]
    w_b_upd = w_b / w_norm
    return w_upd, w_b_upd


def prox_grad_upd_w_conv(w, w_b, regularize_bias=False):
    """
    :param w: shape [N, in_dim, kernel, kernel]
    """
    assert len(w.shape) == 4, "Weight dimension has to be 4 for convolutional prox." \
                              " Got %d dimensions for w_k" % (len(w.shape))
    if regularize_bias:
        tmp_w = torch.cat([w.view(w.size(0), -1), w_b[:, None]], dim=1)  # [N, input_dim + 1]
    else:
        tmp_w = w.view(w.size(0), -1)
    w_norm = torch.linalg.vector_norm(tmp_w, dim=1, ord=2)  # [N,]
    w_norm = torch.clip(w_norm, min=eps)
    w_k_upd = w / w_norm[:, None, None, None]
    w_b_upd = w_b / w_norm
    return w_k_upd, w_b_upd


def regularize(model, thr, regularize_bias=False):
    for idx, grouped_layer in enumerate(model.grouped_layers):
        w = grouped_layer[0].weight.data
        w_b = grouped_layer[0].bias.data
        v = grouped_layer[1].weight.data
        if isinstance(grouped_layer[0], nn.Linear) and isinstance(grouped_layer[1], nn.Linear):
            w_upd, w_b_upd = prox_grad_upd_w(w, w_b, regularize_bias)
            v_upd = prox_grad_upd_v(v, thr)
        elif isinstance(grouped_layer[0], nn.Conv2d) and isinstance(grouped_layer[1], nn.Conv2d):
            w_upd, w_b_upd = prox_grad_upd_w_conv(w, w_b, regularize_bias)
            v_upd = prox_grad_upd_v_conv(v, thr)
        grouped_layer[0].weight.data = w_upd.data
        grouped_layer[0].bias.data = w_b_upd.data
        grouped_layer[1].weight.data = v_upd.data

    return model


def gmean(input_x, dim):
    input_x = torch.clip(input_x, min=eps)
    log_x = torch.log(input_x)
    return torch.exp(torch.mean(log_x, dim=dim))


def layerwise_balance(model):
    pn_list = []
    for grouped_layer in model.grouped_layers:
        pn = get_path_norm(grouped_layer, 2, 2, include_bias=False)
        pn_list.append(pn.sum())
    pn_array = torch.tensor(pn_list)  # [n_group]
    pn_mean = gmean(pn_array, dim=0)  # []
    for idx, grouped_layer in enumerate(model.grouped_layers):
        tmp = torch.clip(pn_list[idx], min=eps)
        scale = pn_mean / tmp
        v = grouped_layer[1].weight.data  # [output_dim, N]
        v_upd = v * scale
        grouped_layer[1].weight.data = v_upd.data
        try:
            v_b = grouped_layer[1].bias.data  # [output_dim,]
            v_b_upd = v_b * scale
            grouped_layer[1].bias.data = v_b_upd.data
        except:
            continue
        try:
            w_b = model.grouped_layers[idx + 1][0].bias.data
            w_b_upd = w_b * scale
            model.grouped_layers[idx + 1][0].bias.data = w_b_upd.data
        except:
            continue
        try:
            model.grouped_layers[idx + 1][1].weight.data = model.grouped_layers[idx + 1][1].weight.data / scale
            pn_list[idx + 1] /= torch.clip(scale, min=eps)
        except:
            continue
    return model

