from typing import Tuple, Dict, List, Union, Any
import os
import dill
from argparse import Namespace
from logging import Logger
import random

import pandas as pd

import torch
import torch.nn as nn
from grover.data import StandardScaler
from grover.util.utils import makedirs, save_checkpoint, load_checkpoint
from grover.util.utils import get_class_sizes, get_data, get_task_names

from utils.model_utils import FitModule
from utils.utils import scoring, BBB_likeness, confuse_matrix, download_and_save_models, un_zip
from utils.dataset_utils import load_MoleculeDataset
from utils.uncertainty_utils import *

from benchmark_ml import load_train_data_for_uncertainty, load_test_data_for_uncertainty, predict, predict_Ttimes
import pickle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def run_active_training_af(args: Namespace, logger: Logger = None) -> List[float]:
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
        x_train, y_train, train_data, train_feature_dicts, scaler, train_data_new = load_train_data_for_uncertainty(args.data_path, prop='PCP')
        x_val, y_val, val_data, val_feature_dicts, val_data_new = load_test_data_for_uncertainty(args.separate_test_path, prop='PCP', scaler=scaler)

        smiles_train = train_data.iloc[:, 0].values
        smiles_val = val_data.iloc[:, 0].values

        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        model_path = os.path.join(save_dir, f'model.pt')

        if not os.path.isfile(model_path):
            debug(f'Download models....')
            download_and_save_models('BBBp_results.zip')
            debug(f'unzip files....')
            un_zip('BBBp_results.zip')

        debug(f'Loading model {model_idx} from {model_path}')
        model = torch.load(model_path)
        model.cuda()

        # scoring
        tasks = ['BBB']
        loss_function = [nn.CrossEntropyLoss() for i in range(len(tasks))]
        val_loss, val_y_list, val_y_pred_list = predict(model,
                                                        val_data,
                                                        val_feature_dicts,
                                                        loss_function)

        train_loss, train_y_list, train_y_pred_list = predict(model,
                                                        train_data,
                                                        train_feature_dicts,
                                                        loss_function)

        val_data['preds'] = val_y_pred_list[0]
        val_data.to_csv(os.path.join(args.save_dir, f'prediction_test_model_{model_idx}.csv'), index=False)

        preds_val = np.array([[1-pred, pred] for pred in val_y_pred_list[0]])
        proba_vote_list.append(preds_val)

        train_score = scoring(train_y_list[0], train_y_pred_list[0], dataset_type=args.dataset_type)
        val_score = scoring(val_y_list[0], val_y_pred_list[0], dataset_type=args.dataset_type)

        info('Finished model building.')
        info(f'{"":^18}\t{"Training":^8}\t{"Testing":^8}')
        for key in train_score.keys():
            info(f'{key:^18}\t {train_score[key]:.4f}  \t {val_score[key]:.4f}  ')

        # save_latent_feats
        val_features = AttentiveFP_featurize_loader(model, val_data, val_feature_dicts, args)
        dill.dump((smiles_val, val_features, val_y_list[0]),
                  open(os.path.join(args.save_dir, f'latent_feature_test_model_{model_idx}.pkl'), 'wb'))

        train_features = AttentiveFP_featurize_loader(model, train_data, train_feature_dicts, args)
        dill.dump((smiles_train, train_features, train_y_list[0]),
                  open(os.path.join(args.save_dir, f'latent_feature_model_train_{model_idx}.pkl'), 'wb'))

        # # latent_distance
        LatentDist_u = AttentiveFP_latent_distance(model, train_data, train_feature_dicts, val_data, val_feature_dicts, args)

        # entropy_uncertainty
        Entropy_u = entropy_uncertainty(preds_val)

        # # MCdropout_uncertainty
        mc_pred_probas = predict_Ttimes(model,
                                        val_data,
                                        val_feature_dicts,
                                        loss_function,
                                        T=args.pred_times)
        MCdropout_u = mc_dropout(mc_pred_probas)

        uncertainty_list = [LatentDist_u, Entropy_u, MCdropout_u]
        uncertainty_list_sum = [uncertainty_list_sum[i] + uc for i, uc in enumerate(uncertainty_list)]

    ensemble_pred_val = np.sum(np.array(proba_vote_list)[:, :, 1], axis=0) / args.ensemble_size
    ensemble_val_score = scoring(val_y_list[0], ensemble_pred_val, dataset_type=args.dataset_type)
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
    num_tasks = 1
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
            [np.squeeze(np.array(val_y_list[0]))],
            np.array(uncertainty_list)]
    ).T
    test_result = pd.DataFrame(data, columns=ind)
    test_result.insert(0, 'SMILES', smiles_val)
    props_val = np.array([BBB_likeness(smiles) for smiles in smiles_val]).T
    for i, prop in enumerate(['MW', 'PSA', 'HBA', 'HBD', 'ROTB', 'ALOGP']):
        test_result.insert(i+1, prop, props_val[i])
    test_result.insert(7, 'Confuse', [confuse_matrix(true, pred) for pred, true in zip(ensemble_pred_val, val_y_list[0])])
    test_result.to_csv(os.path.join(args.save_dir, f'test_result.csv'), index=False)
    info(f'Predictions and uncertainties saved in {os.path.join(args.save_dir, f"test_result.csv")}.')

