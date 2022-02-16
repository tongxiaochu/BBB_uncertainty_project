import random

import numpy as np
import torch
from rdkit import RDLogger

from grover.util.parsing import parse_args, get_newest_train_args
from grover.util.utils import create_logger




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

    args = parse_args()
    model_dir = args.save_dir.split('/')[-1]
    if 'mlp' in model_dir:
        from task.train_ml import run_active_training
    elif 'rf' in model_dir:
        from task.train_ml import run_active_training
    elif 'AttentiveFP' in model_dir:
        from task.train_attentivefp import run_active_training
    else:
        from task.train import run_active_training

    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)

    run_active_training(args, logger)


