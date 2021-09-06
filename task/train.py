from typing import Tuple, Dict, List, Union, Any
import os
import dill
from argparse import Namespace
from logging import Logger
import random

import pandas as pd

import torch

from grover.data import StandardScaler
from grover.util.utils import makedirs, save_checkpoint, load_checkpoint
from grover.util.utils import get_class_sizes, get_data, get_task_names

from utils.model_utils import FitModule
from utils.utils import scoring, BBB_likeness, confuse_matrix
from utils.dataset_utils import load_MoleculeDataset
from utils.uncertainty_utils import *


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def run_active_training(args: Namespace, logger: Logger = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """
    setup_seed(args.seed)
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print
    # pin GPU to local rank.
    idx = args.gpu
    if args.gpu is not None:
        torch.cuda.set_device(idx)
    info(args)

    # Get data
    debug('Loading data')
    args.task_names = get_task_names(args.data_path)
    train_data = get_data(path=args.data_path, args=args, logger=logger)
    
    if train_data.data[0].features is not None:
        args.features_dim = len(train_data.data[0].features)
    else:
        args.features_dim = 0

    args.num_tasks = train_data.num_tasks()
    args.features_size = train_data.features_size()
    debug(f'Number of tasks = {args.num_tasks}')

    val_data = get_data(path=args.separate_test_path, features_path=args.separate_test_features_path, args=args, logger=logger)

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    args.train_data_size = len(train_data)

    info(f'train size = {len(train_data):,} | val size = {len(val_data):,} ')
    if args.dataset_type == 'classification':
        info('Class sizes')
        for data in [train_data, val_data]:
            class_sizes = get_class_sizes(data)
            for i, task_class_sizes in enumerate(class_sizes):
                info(f'{args.task_names[i]} '
                      f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    # Train ensemble of models
    uncertainty_list_sum = [0, 0, 0, 0, 0, 0, 0]
    proba_vote_list = []
    for model_idx in range(args.ensemble_size):

        # Load data
        smiles_train, x_train, y_train, mask_train = load_MoleculeDataset(train_data, args)
        smiles_val, x_val, y_val, mask_val = load_MoleculeDataset(val_data, args)

        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        model_path = os.path.join(save_dir, f'model.ckpt')
        if os.path.isfile(model_path):
            debug(f'Loading model {model_idx} from {model_path}')
            network = load_checkpoint(model_path, cuda=args.cuda, current_args=args, logger=logger)
            model = FitModule(model_idx, None, args, logger)
            model.model = network
        else:
            raise (f'There is no finetuned model in {model_path}')

        # save_latent_feats
        features = featurize_loader(network, x_val, args)
        dill.dump((smiles_val, features, y_val), open(os.path.join(args.save_dir, f'latent_feature_test_model_{model_idx}.pkl'),'wb'))

        # scoring
        preds_val = model.predict_proba(x_val)
        proba_vote_list.append(preds_val)
        hard_preds_val = preds_val[:, 1].reshape(-1, 1)

        train_score = scoring(y_train, model.predict(x_train), dataset_type=args.dataset_type)
        val_score = scoring(y_val, hard_preds_val, dataset_type=args.dataset_type)

        info('Finished model building.')
        info(f'{"":^18}\t{"Training":^8}\t{"Testing":^8}')
        for key in train_score.keys():
            info(f'{key:^18}\t {train_score[key]:.4f}  \t {val_score[key]:.4f}  ')

        # latent_distance
        LatentDist_u = latent_distance(network, x_train, x_val, args)
        # entropy_uncertainty
        Entropy_u = entropy_uncertainty(preds_val)
        # MCdropout_uncertainty
        mc_pred_probas = model.predict_Ttimes(x_val, T=args.pred_times)
        MCdropout_u = mc_dropout(mc_pred_probas)
        
        uncertainty_list = [LatentDist_u, Entropy_u, MCdropout_u]
        uncertainty_list_sum = [uncertainty_list_sum[i] + uc for i, uc in enumerate(uncertainty_list)]

    ensemble_pred_val = np.sum(np.array(proba_vote_list)[:, :, 1], axis=0) / args.ensemble_size
    ensemble_val_score = scoring(y_val, ensemble_pred_val, dataset_type=args.dataset_type)
    info('Finished model building.')
    info(f'{"":^18}\t{"Ensemble Testing"}')
    for key in ensemble_val_score.keys():
        info(f'{key:^18}\t  {ensemble_val_score[key]:.4f}  ')

    # fingerprint_distance
    FPsDist_u = fingerprint_distance(smiles_train, smiles_val, fp='ecfp')
    # multi-initial uncertainty
    Multi_init_u = multi_initial(np.array(proba_vote_list)[:, :, 1])

    uncertainty_list = [FPsDist_u] + [uc / 5 for uc in uncertainty_list_sum] + [Multi_init_u]

    # write results to csv
    ind = [
    ['Prediction'] * args.num_tasks + \
    ['Target'] * args.num_tasks + \
    ['FPsDist'] * args.num_tasks + \
    ['LatentDist'] * args.num_tasks + \
    ['Entropy'] * args.num_tasks + \
    ['MC-dropout'] * args.num_tasks + \
    ['Multi-initial'] * args.num_tasks
    ]
    ind = pd.MultiIndex.from_tuples(list(zip(*ind)))
    data = np.concatenate(
        [
            [np.squeeze(np.array(ensemble_pred_val))],
            [np.squeeze(np.array(y_val))],
            np.array(uncertainty_list)]
    ).T
    test_result = pd.DataFrame(data, columns=ind)
    test_result.insert(0, 'SMILES', smiles_val)
    props_val = np.array([BBB_likeness(smiles) for smiles in smiles_val]).T
    for i, prop in enumerate(['MW', 'PSA', 'HBA', 'HBD', 'ROTB', 'ALOGP']):
        test_result.insert(i+1, prop, props_val[i])
    test_result.insert(7, 'Confuse', [confuse_matrix(true, pred) for pred, true in zip(ensemble_pred_val, y_val)])
    test_result.to_csv(os.path.join(args.save_dir, f'test_result.csv'), index=False)
    info(f'Predictions and uncertainties saved in {os.path.join(args.save_dir, f"test_result.csv")}.')

