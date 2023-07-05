import pprint
import torch
import wandb
import os

from args_helper import args
from logger import setup_logger
from regularize import trainer as regularize_trainer
from utils import set_seed, get_dataset, get_model, get_optimizer, get_scheduler, update_iter_and_epochs, get_criterion


def main():
    logger = setup_logger(name=args.logger_name, args=args)
    logger.info("Call with args: \n{}".format(pprint.pformat(vars(args))))
    project_name = "{}_{}".format(args.which_dataset.lower(), args.arch.lower())
    wandb.init(project=project_name, name=args.logger_name, config=vars(args))
    set_seed(args.seed, logger)

    if args.cuda:
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cpu")
        torch.set_num_threads(4)
    logger.info("Using device {}".format(device))
    dataset = get_dataset(args=args, logger=logger)
    update_iter_and_epochs(dataset=dataset, args=args, logger=logger)
    model = get_model(args=args, logger=logger, dataset=dataset).to(device)
    criterion = get_criterion(criterion_type=args.criterion)
    optimizer = get_optimizer(args=args, model=model)
    scheduler = get_scheduler(optimizer=optimizer, logger=logger, args=args)

    regularize_trainer(
        dataset=dataset, device=device, model=model,
        args=args, optimizer=optimizer, scheduler=scheduler, criterion=criterion, wandb=wandb)


if __name__ == "__main__":
    main()
