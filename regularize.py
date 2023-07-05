import torch
import tqdm
import os

from utils import test, calc_nonzero_neuron
from regularize_func import regularize, layerwise_balance, collect_other_norm, collect_grouped_norm


def test_and_log(model, regularize_bias, dataset, criterion, device, result_dict):
    # dataset
    train_loader = dataset.train_loader
    val_loader = dataset.val_loader
    test_loader = dataset.test_loader

    test_acc, test_loss = test(model, test_loader, device, record_loss=True, criterion=criterion)
    result_dict['acc']['test'].append(test_acc)
    result_dict['loss']['test_loss'].append(test_loss)
    val_acc, val_loss = test(model, val_loader, device, record_loss=True, criterion=criterion)
    result_dict['acc']['val'].append(val_acc)
    result_dict['loss']['val_loss'].append(val_loss)
    train_acc, train_loss = test(model, train_loader, device, record_loss=True, criterion=criterion)
    result_dict['acc']['train'].append(train_acc)
    result_dict['loss']['train_loss'].append(train_loss)

    # calculate sparsity
    total_nz, total_pn, _ = calc_nonzero_neuron(model, regularize_bias)
    result_dict['act']['total_nz'].append(total_nz)
    result_dict['loss']['total_pn'].append(total_pn)

    wandb_dict = dict({
        "train acc": train_acc,
        "train loss": train_loss,
        "val acc": val_acc,
        "val loss": val_loss,
        "test acc": test_acc,
        "test loss": test_loss,
        "active neurons (%)": total_nz,
        "l2 path norm": total_pn,
    })

    return result_dict, wandb_dict


def wandb_log(wandb, wandb_dict, idx_iter, optimizer):
    wandb_dict['idx_iter'] = idx_iter
    wandb_dict['lr'] = optimizer.param_groups[0]["lr"]
    if wandb is not None:
        wandb.log(wandb_dict)


def init_result_dict():
    acc_dict, act_dict, loss_dict = {}, {}, {}
    acc_dict['train'], acc_dict['test'], acc_dict['val'] = [], [], []
    act_dict['total_nz'] = []
    loss_dict['total_pn'], loss_dict['train_loss'], loss_dict['val_loss'], loss_dict['test_loss'] = [], [], [], []
    result_dict = {'acc': acc_dict, 'act': act_dict, 'loss': loss_dict}
    return result_dict


def trainer(dataset, device, model, args, optimizer, scheduler, criterion, wandb):

    # use args:
    total_iter = args.total_iter
    total_epoch = args.total_epoch
    log_freq = args.log_freq
    save_freq = args.save_freq
    wd_param = args.wd_param
    dest_dir = args.dest_dir
    flag_with_loss_term = args.with_loss_term
    flag_with_prox_upd = args.with_prox_upd
    regularize_bias = args.regularize_bias

    result_dict = init_result_dict()

    # start training
    idx_iter = 0
    flag_iter = False  # flag of whether it has reached the total number of iterations

    # begin training
    for idx_epoch in tqdm.tqdm(range(total_epoch + 1)):
        if flag_iter:  # has reached the total number of iterations
            break
        else:
            # Training
            for batch_idx, (imgs, targets) in enumerate(dataset.train_loader):
                model.train()
                # Frequency for Testing
                if idx_iter % log_freq == 0:
                    with torch.no_grad():
                        result_dict, wandb_dict = test_and_log(
                            model, regularize_bias, dataset, criterion, device, result_dict)

                    PATH_result = os.path.join(dest_dir, "result.pt")
                    torch.save(result_dict, PATH_result)
                    wandb_log(wandb, wandb_dict, idx_iter, optimizer)
                # Frequency for Saving
                if idx_iter % save_freq == 0:
                    PATH_model = os.path.join(dest_dir, "model_idx_{}_acc_{}_act_{}".format(
                        idx_iter, round(wandb_dict['test acc'], 2), int(wandb_dict['active neurons (%)'])
                    ).replace(".", "_") + ".pt")
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, PATH_model)

                # Training
                imgs, targets = imgs.to(device), targets.to(device)
                output = model(imgs)
                train_loss = criterion(output, targets)
                other_l2_norm = collect_other_norm(model, wd_param, regularize_bias)
                if flag_with_loss_term:
                    l2_norm = collect_grouped_norm(model, wd_param, "wd", regularize_bias)
                else:
                    l2_norm = 0.
                (train_loss + l2_norm + other_l2_norm).backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if flag_with_prox_upd:
                    if not regularize_bias:
                        model = layerwise_balance(model)

                    actual_wd_param = wd_param * optimizer.param_groups[0]["lr"]
                    model = regularize(model, actual_wd_param, regularize_bias)

                idx_iter += 1
                if idx_iter == total_iter:
                    flag_iter = True
                    break

            if scheduler is not None:
                scheduler.step()

    return model, result_dict
