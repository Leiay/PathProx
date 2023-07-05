import argparse
import yaml
import sys

from configs import parser as _parser

global args


class ArgsHelper:
    def parse_arguments(self):
        parser = argparse.ArgumentParser()

        # ============================================================================================================ #
        # system related
        parser.add_argument(
            "--cuda",
            action="store_true",
            required=True,
            help="[system] bool value to use gpu or not"
        )
        parser.add_argument(
            "--gpu",
            type=int,
            default=0,
            help="[system] Override the default choice for a CUDA-enabled GPU by specifying the GPU\"s integer index "
                 "(i.e. \"0\" for \"cuda:0\") "
        )
        parser.add_argument(
            "--num-workers",
            type=int,
            default=4,
            help="[system] Number of workers"
        )
        parser.add_argument(
            "--logger-name",
            type=str,
            required=True,
            help="[system] logger name needs to be specified"
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="[system] Set seed for the program"
        )
        parser.add_argument(
            "--dest-dir",
            type=str,
            default=None,
            help="[system] result destination directory, will be overwritten later"
        )
        parser.add_argument(
            "--config",
            default=None,
            help="Config file to use"
        )
        # ============================================================================================================ #
        # data related
        parser.add_argument(
            "--data-path",
            default="data/",
            help="[dataset] path to dataset base directory"
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=64,
            help="[dataset] Input batch size for training (default: 64)"
        )
        parser.add_argument(
            "--which-dataset",
            type=str,
            default="mnist",
            help="[dataset] Dataset to train the model with. Can be CIFAR10 or mnist (default: mnist)"
        )
        # ============================================================================================================ #
        # optimizer related
        parser.add_argument(
            "--optimizer",
            type=str,
            default="SGD",
            help="[optimizer] optimizer, choice  SGD | Adam | AdamW"
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=0.01,
            help="[optimizer] learning rate"
        )
        parser.add_argument(
            "--momentum",
            type=float,
            default=0.9,
            help="[optimizer] momentum"
        )
        # ============================================================================================================ #
        # criterion related
        parser.add_argument(
            "--criterion",
            type=str,
            default="CE",
            help="[optimizer] optimizer, choice  CE | MSE"
        )
        # ============================================================================================================ #
        # scheduler related
        parser.add_argument(
            "--lr-scheduler",
            type=str,
            default=None,
            help="[scheduler] learning rate scheduler"
        )
        parser.add_argument(
            "--gamma",
            type=float,
            default=None,
            help="[scheduler] learning rate decay"
        )
        parser.add_argument(
            "--milestone",
            type=int,
            nargs="+",
            default=[],
            help="[scheduler] milestone for multi-step learning rate scheduler"
        )
        # ============================================================================================================ #
        # model related
        parser.add_argument(
            "--arch",
            type=str,
            default="[model] architecture name, choice: ",
        )
        parser.add_argument(
            "--num-hidden",
            type=int,
            default=None,
            help="[model] indicate number of hidden neurons "
        )
        # training related
        parser.add_argument(
            "--total-iter",
            type=int,
            default=0,
            help="[train] total number of iterations to run"
        )
        parser.add_argument(
            "--total-epoch",
            type=int,
            default=0,
            help="[train] total number of iterations to run"
        )
        parser.add_argument(
            "--log-freq",
            type=int,
            help="[train] frequency to print out the test and val information"
        )
        parser.add_argument(
            "--save-freq",
            type=int,
            help="[train] frequency to save the model and loss information"
        )
        # ============================================================================================================ #
        # regularize related
        parser.add_argument(
            "--wd-param",
            type=float,
            help="[hyperparameter] the weight decay parameter for training"
        )
        parser.add_argument(
            "--with-loss-term",
            action="store_true",
            default=False,
            help="[regularize] if set true, regularize with weight decay term add to loss"
        )
        parser.add_argument(
            "--with-prox-upd",
            action="store_true",
            default=False,
            help="[regularize] if set true, regularize the coupling layers with path norm, "
                 "the other part with weight decay term add to loss"
        )
        parser.add_argument(
            "--regularize-bias",
            action="store_true",
            default=False,
            help="[regularize] if set to true, will also regularize the bias"
        )
        # ============================================================================================================ #
        args = parser.parse_args()
        self.get_config(args)
        return args

    def get_config(self, parser_args):
        # get commands from command line
        override_args = _parser.argv_to_vars(sys.argv)

        # load yaml file
        if parser_args.config is not None:
            yaml_txt = open(parser_args.config).read()

            # override args
            loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
            for v in override_args:
                loaded_yaml[v] = getattr(parser_args, v)

            print(f"=> Reading YAML config from {parser_args.config}")
            parser_args.__dict__.update(loaded_yaml)

    def get_args(self):
        global args
        args = self.parse_arguments()
        from utils import set_dest_dir
        if args.dest_dir is None:
            set_dest_dir(args)


argshelper = ArgsHelper()
argshelper.get_args()
