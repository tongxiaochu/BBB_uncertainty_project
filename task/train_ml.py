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
from utils.utils import scoring, BBB_likeness, confuse_matrix, MLP_featurize_loader, download_and_save_models, un_zip
from utils.dataset_utils import load_MoleculeDataset
from utils.uncertainty_utils import *
from benchmark_ml import load_train_data_for_uncertainty, load_test_data_for_uncertainty

import pickle

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def run_active_training_ml(args: Namespace, logger: Logger = None) -> List[float]:
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

    info(args)

    # Train ensemble of models
    uncertainty_list_sum = [0, 0, 0, 0, 0, 0, 0]
    proba_vote_list = []
    for model_idx in range(args.ensemble_size):

        # Load data
        debug('Loading data.....')
        x_train, y_train, train_data, train_feature_dicts, scaler, train_data_new = load_train_data_for_uncertainty(args.data_path, prop=args.feature_type)
        x_val, y_val, val_data, val_feature_dicts, val_data_new = load_test_data_for_uncertainty(args.separate_test_path, prop=args.feature_type, scaler=scaler)

        smiles_train = train_data_new.iloc[:, 0].values
        smiles_val = val_data_new.iloc[:, 0].values

        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        model_path = os.path.join(save_dir, f'model.ckpt')
        if not os.path.isfile(model_path):
            debug(f'Download models....')
            download_and_save_models('BBBp_results.zip')
            debug(f'unzip files....')
            un_zip('BBBp_results.zip')

        debug(f'Loading model {model_idx} from {model_path}')
        with open(model_path, 'rb') as f:
            estimator = pickle.load(f)

        # save_latent_feats
        if args.parser_name == 'MLP':
            features = MLP_featurize_loader(x_val, estimator, layer=0)
            dill.dump((smiles_val, features, y_val), open(os.path.join(args.save_dir, f'latent_feature_test_model_{model_idx}.pkl'), 'wb'))

            train_features = MLP_featurize_loader(x_train, estimator, layer=0)
            dill.dump((smiles_train, train_features, y_train), open(os.path.join(args.save_dir, f'latent_feature_model_train_{model_idx}.pkl'), 'wb'))

        # scoring
        preds_val = estimator.predict_proba(x_val)
        proba_vote_list.append(preds_val)
        hard_preds_val = preds_val[:, 1].reshape(-1, 1)
        train_score = scoring(y_train, estimator.predict_proba(x_train)[:, 1], dataset_type=args.dataset_type)
        val_score = scoring(y_val, hard_preds_val, dataset_type=args.dataset_type)

        info('Finished model building.')
        info(f'{"":^18}\t{"Training":^8}\t{"Testing":^8}')
        for key in train_score.keys():
            info(f'{key:^18}\t {train_score[key]:.4f}  \t {val_score[key]:.4f}  ')

        #latent_distance
        if args.parser_name == 'MLP':
            LatentDist_u = MLP_latent_distance(estimator, x_train, x_val, args)
        else:
            LatentDist_u = np.zeros_like(y_val)

        # entropy_uncertainty
        Entropy_u = entropy_uncertainty(preds_val)

        # # MCdropout_uncertainty
        MCdropout_u = np.zeros_like(y_val)

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

    num_tasks = 1
    # write results to csv
    ind = [
    ['Prediction'] * num_tasks + \
    ['Target'] * num_tasks + \
    ['FPsDist'] * num_tasks + \
    ['LatentDist'] * num_tasks + \
    ['Entropy'] * num_tasks + \
    ['MC-dropout'] * num_tasks + \
    ['Multi-initial'] * num_tasks
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


