import random

import numpy as np
import torch
from rdkit import RDLogger

from argparse import ArgumentParser

from grover.util.parsing import add_GROVER_args, modify_train_args, add_AttentiveFP_args, add_RF_args, add_MLP_args#parse_args, get_newest_train_args
from grover.util.utils import create_logger

from task.train_ml import run_active_training_ml
from task.train_attentivefp import run_active_training_af
from task.train import run_active_training_GROVER



def setup(seed):
    # frozen random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # setup random seed
    setup(seed=1234)
    # supress rdkit logger
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    parser = ArgumentParser()
    subparser = parser.add_subparsers(title='subcommands',
                                    dest='parser_name',
                                    help='Subcommands for BBBp prediction by GROVER, AttentiveFP, MLP, RF')
    parser_GROVER = subparser.add_parser('GROVER', help='predict BBBp by GROVER')
    add_GROVER_args(parser_GROVER)
    parser_AttentiveFP = subparser.add_parser('AttentiveFP', help='predict BBBp by AttentiveFP')
    add_AttentiveFP_args(parser_AttentiveFP)
    parser_AttentiveFP = subparser.add_parser('RF', help='predict BBBp by RF')
    add_RF_args(parser_AttentiveFP)
    parser_MLP = subparser.add_parser('MLP', help='predict BBBp by MLP')
    add_MLP_args(parser_MLP)

    args = parser.parse_args()

    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)

    if args.parser_name == 'GROVER':
        modify_train_args(args)
        run_active_training_GROVER(args, logger)
    elif args.parser_name == 'AttentiveFP':
        print('predict by AttentiveFP')
        run_active_training_af(args, logger)
    else:
        print('BBBp prediction by ML models')
        run_active_training_ml(args, logger)




