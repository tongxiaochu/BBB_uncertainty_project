"""
Uncertainty measures and uncertainty based sampling strategies for the active learning models.
"""
from typing import Tuple, Dict, List, Union, Any
import os
import torch
from torch import nn
import argparse
from collections import OrderedDict

import numpy as np
from tqdm import tqdm

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import QED
from descriptastorus.descriptors import rdNormalizedDescriptors

from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
from grover.util.metrics import get_metric_func

from utils.model_utils import MidFeature
from utils.dataset_utils import MyDataset, GraphCollator


def scoring(y: np.array, y_pred: np.array, dataset_type: str, metrics_func: Union[List, str] = 'default') -> Dict:
    if metrics_func == 'default':
        if dataset_type == 'classification':
            metrics_func = ['roc_auc', 'matthews_corrcoef', 'recall', 'precision', 'specificity', 'prc_auc', 'balanced_accuracy']
        elif dataset_type == 'regression':
            metrics_func = ['rmse', 'mae', 'r2', 'pearsonr']
    else:
        if not isinstance(metrics_func, List):
            metrics_func = [metrics_func]
    return OrderedDict({m: get_metric_func(m)(y, y_pred) for m in metrics_func})


def get_fps(smiles, fp='ecfp'):
    if isinstance(smiles, str):
        MolSuppl = [Chem.MolFromSmiles(i) for i in [smiles]]
    else:
        MolSuppl = [Chem.MolFromSmiles(i) for i in smiles]
    if fp == 'ecfp':
        fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 3, 4096) for mol in MolSuppl]
    elif fp == 'maccs':
        fps = [AllChem.GetMACCSKeysFingerprint(mol) for mol in MolSuppl]
    elif fp == 'topo':
        fps = [FingerprintMols.FingerprintMol(mol) for mol in MolSuppl]
    elif fp == 'avalon':
        fps = [pyAvalonTools.GetAvalonFP(mol) for mol in MolSuppl]
    elif fp == 'rdkit_2d_normalized':
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        fps = [generator.process(mol) for mol in MolSuppl]
    return fps


def standardized_euclidean_distance(support_vectors, query_vectors):
    support_vectors = preprocessing.scale(support_vectors, axis=0)
    query_vectors = preprocessing.scale(query_vectors, axis=0)
    eu_dist = []
    for qv in tqdm(query_vectors):
        d = [np.sqrt(np.sum(np.square(qv - sv))) for sv in support_vectors]
        eu_dist.append(d)
    return eu_dist


def tanimoto_distance(fp_support, fp_queries):
    ta_dist = []
    for fp in tqdm(fp_queries):
        ta_dist .append([1 - s for s in DataStructs.BulkTanimotoSimilarity(fp, fp_support)])
    return ta_dist


def featurize_loader(network, X, args):
    if args.cuda:
        network = network.cuda()
    y = torch.zeros(X.shape[0])
    masks = torch.zeros(X.shape[0])
    data = MyDataset(X, y, masks)
    loader = DataLoader(data, args.batch_size, False, collate_fn=GraphCollator(args))
    feat_model = MidFeature(network)
    feat_model.eval()
    features = []
    for i, item in enumerate(loader):
        batch, features_batch, targets, mask = item
        with torch.no_grad():
            feature = feat_model(batch, features_batch)
        features.append(feature)

    return torch.cat(features).squeeze().cpu()


def rdkit_2d_normalized(smiles):
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    features = generator.process(smiles)[1:]
    return features


def BBB_likeness(smiles):
    """
    molecular weight <500;
    60 < total polar surface area < 90;
    number of hydrogen-bond acceptors < 10;
    number of hydrogen-bond donors < 5;
    number of rotatable bonds < 10;
    1.5 < logP < 2.5
    """
    mol = Chem.MolFromSmiles(smiles)
    props = QED.properties(mol)
    MW = props.MW
    PSA = props.PSA
    HBA = props.HBA
    HBD = props.HBD
    ROTB = props.ROTB
    ALOGP = props.ALOGP
    return [MW, PSA, HBA, HBD, ROTB, ALOGP]


def confuse_matrix(y, y_pred):
    if y > 0.5 and y_pred > 0.5:
        return 'TP'
    if y > 0.5 and y_pred <= 0.5:
        return 'FN'
    if y < 0.5 and y_pred > 0.5:
        return 'FP'
    if y < 0.5 and y_pred <= 0.5:
        return 'TN'