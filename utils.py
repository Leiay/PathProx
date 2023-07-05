import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import datetime
import torch
import math
import os

eps = 1e-12


def log_message(logger, message):
    if logger is None:
        print(message)
    else:
        logger.info(message)


def get_path_norm(grouped_layer, w_norm_deg=2, v_norm_deg=2, requires_grad=False, include_bias=False):
    w = grouped_layer[0].weight if requires_grad else grouped_layer[0].weight.data  # [N, input_dim]
    if include_bias:
        w_b = grouped_layer[0].bias if requires_grad else grouped_layer[0].bias.data  # [N,]

    v = grouped_layer[1].weight if requires_grad else grouped_layer[1].weight.data  # [output_dim, N]

    if isinstance(grouped_layer[0], nn.Linear) and isinstance(grouped_layer[1], nn.Linear):
        if include_bias:
            w = torch.cat([w, w_b[:, None]], dim=1)  # [N, input_dim + 1]
        w_norm = torch.linalg.vector_norm(w, dim=1, ord=w_norm_deg)
        v_norm = torch.linalg.vector_norm(v, dim=0, ord=v_norm_deg)
        path_norm = w_norm * v_norm
    elif isinstance(grouped_layer[0], nn.Conv2d) and isinstance(grouped_layer[1], nn.Conv2d):
        if include_bias:
            # w [N, input_dim, k, k] -> [N, input_dim * k * k], w_b [N,]
            w = torch.cat([w.view(w.size(0), -1), w_b[:, None]], dim=1)  # [N, input_dim + 1]
        else:
            w = w.view(w.size(0), -1)
        w_norm = torch.linalg.vector_norm(w, dim=1, ord=w_norm_deg)
        v_norm = torch.linalg.vector_norm(v, dim=(0, 2, 3), ord=v_norm_deg)
        path_norm = w_norm * v_norm
    else:
        raise ValueError("Wrong layers passed into path norm implementation.")

    # if more:
    #     return path_norm, w_norm, v_norm
    return path_norm


def update_iter_and_epochs(dataset, args, logger):
    per_epoch_iter = math.ceil(len(dataset.train_loader.dataset) // args.batch_size)  # number of iterations per epoch
    if args.total_epoch == 0:
        args.total_epoch = args.total_iter // per_epoch_iter + 1
    elif args.total_iter == 0:
        args.total_iter = args.total_epoch * per_epoch_iter
    else:
        if not args.lr_rewind:
            assert args.total_iter == args.total_epoch * per_epoch_iter
        else:
            logger.info("Since it is learning rate rewinding, use self-defined total_iter and total_epochs")
    logger.info("Update the total iter to be {}, and total epoch to be {}".format(args.total_iter, args.total_epoch))


def get_dataset(args, logger):
    import dataset
    message = "=> Getting {} dataset".format(args.which_dataset)
    log_message(logger, message)
    dataset = getattr(dataset, args.which_dataset)(args)
    return dataset


def get_criterion(criterion_type):
    if criterion_type.lower() == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
    elif criterion_type.lower() == 'mse':
        criterion = lambda input, target: F.mse_loss(input.squeeze(), target.squeeze(), reduction="mean")
    else:
        raise NotImplementedError("only support criterion CE (cross entropy) | MSE (mean squared error)")
    return criterion


def get_model(args, logger, dataset):
    import models
    message = "=> Creating model {}".format(args.arch)
    log_message(logger, message)
    if "mlp" in args.arch.lower():
        model = models.__dict__[args.arch](
            input_dim=dataset.input_dim if dataset.input_channel is None else (dataset.input_dim ** 2) * dataset.input_channel,
            num_hidden=args.num_hidden, num_classes=dataset.num_classes)
    elif "vgg" in args.arch.lower():
        model = models.__dict__[args.arch](num_classes=dataset.num_classes)
    else:
        model = models.__dict__[args.arch](input_dim=dataset.input_dim, num_classes=dataset.num_classes)
    return model


def get_optimizer(args, model):
    opt_algo = args.optimizer
    lr = args.lr
    mom = args.momentum
    if opt_algo.lower() == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.)
    elif opt_algo.lower() == "sgd":
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=mom, weight_decay=0.)
    elif opt_algo.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.)
    else:
        raise NotImplementedError("Only support Adam, AdamW and SGD")
    return optimizer


def get_scheduler(optimizer, logger, args):
    scheduler = args.lr_scheduler
    max_epochs = args.total_epoch
    if scheduler == 'cosine_lr':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        message = "scheduler: use cosine learning rate decay, with max epochs {}".format(max_epochs)
    elif scheduler == 'exp_lr':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        message = "scheduler: use cosine learning rate decay, with max epochs {}".format(max_epochs)
    elif scheduler == "multi_step":
        gamma = args.gamma
        milestones = args.milestone
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        message = "scheduler: use multistep learning rate decay, with milestones {} and gamma {}".format(milestones, gamma)
    else:
        message = "Policy not specified. Default is None"
        lr_scheduler = None
    log_message(logger, message)

    return lr_scheduler


def set_seed(seed, logger):
    import random
    import os
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    message = "Seeded everything: {}".format(seed)
    log_message(logger, message)


def set_dest_dir(args):
    if not os.path.exists("results"):
        os.mkdir("results")
    subfolder_name = "{}_{}".format(args.which_dataset, args.arch)
    if not os.path.exists("results/{}".format(subfolder_name)):
        os.mkdir("results/{}".format(subfolder_name))
    now = datetime.datetime.now().strftime('%m%d%H%M%S')
    dest_dir = os.path.join("results", subfolder_name, "{}_{}".format(now, args.logger_name))
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    args.dest_dir = dest_dir


def test(model, test_loader, device, record_loss=False, criterion=None):
    model.eval()
    correct = 0
    if record_loss:
        assert criterion is not None
    loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if record_loss:
                test_loss = criterion(output, target)
                # criterion default reduce is 'mean', so we will multiply by batch size to restore the loss sum
                loss += test_loss * target.shape[0]
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        accuracy = 100. * correct / len(test_loader.dataset)
        loss = loss / len(test_loader.dataset)
    if record_loss:
        return accuracy, loss
    return accuracy


def calc_nonzero_neuron(model, regularize_bias=False):
    tot_nz, tot = 0, 0
    grouped_nz = []
    total_pn = 0.
    for idx, grouped_layer in enumerate(model.grouped_layers):
        pn = get_path_norm(grouped_layer, 2, 2, regularize_bias)
        N = pn.shape[0]  # batch size
        total_pn += pn.sum().item()
        nz = (pn > 0).sum()
        grouped_nz.append(nz)
        tot_nz += nz
        tot += N
    return tot_nz / (tot + eps) * 100., total_pn, grouped_nz
