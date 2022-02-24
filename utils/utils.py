"""
Uncertainty measures and uncertainty based sampling strategies for the active learning models.
"""
from typing import Tuple, Dict, List, Union, Any
import os
import torch
from torch import nn
import argparse
from collections import OrderedDict
import zipfile
from contextlib import closing
import requests

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

from sklearn.neural_network._base import ACTIVATIONS
from AttentiveFP import get_smiles_array



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

def AttentiveFP_featurize_loader(network, dataset, feature_dicts, args):

    network = network.cuda()

    network.eval()
    batch_size = 16
    batch_list = []
    valList = np.arange(0, dataset.shape[0])
    features = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i + batch_size]
        batch_list.append(batch)
    for counter, train_batch in enumerate(batch_list):
        batch_df = dataset.iloc[train_batch, :]
        smiles_list = batch_df.cano_smiles.values

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,
                                                                                                     feature_dicts)
        with torch.no_grad():
            _, _, mol_feature = network(torch.Tensor(x_atom).to('cuda'),
                           torch.Tensor(x_bonds).to('cuda'),
                           torch.cuda.LongTensor(x_atom_index),
                           torch.cuda.LongTensor(x_bond_index),
                           torch.Tensor(x_mask).to('cuda')
                           )

            # feature = network.mol_feature
            feature = mol_feature

        features.append(feature)

    return torch.cat(features).squeeze().cpu()

def MLP_featurize_loader(x_data, model, layer=0):
    L = ACTIVATIONS['relu'](np.matmul(x_data, model.coefs_[layer]) + model.intercepts_[layer])
    layer += 1
    if layer >= len(model.coefs_) - 1:
        return L
    else:
        return MLP_featurize_loader(L, model, layer=layer)


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

def download_and_save_models(file_name):
    url = 'https://zenodo.org/record/6253524/files/BBBp_results.zip?download=1'
    with closing(requests.get(url, stream=True)) as response:
        chunk_size = 1024
        content_size = int(response.headers['content-length'])
        progress = ProgressBar(file_name, total=content_size,
                                         unit="KB", chunk_size=chunk_size, run_status="downloading", fin_status="finished")
        with open(file_name, "wb") as file:
           for data in response.iter_content(chunk_size=chunk_size):
               file.write(data)
               progress.refresh(count=len(data))

def un_zip(file_name):
    zip_file = zipfile.ZipFile(file_name)
    for names in zip_file.namelist():
        zip_file.extract(names)
    zip_file.close()

class ProgressBar(object):

    def __init__(self, title,
                 count=0.0,
                 run_status=None,
                 fin_status=None,
                 total=100.0,
                 unit='', sep='/',
                 chunk_size=1.0):
        super(ProgressBar, self).__init__()
        self.info = "[%s]%s %.2f %s %s %.2f %s"
        self.title = title
        self.total = total
        self.count = count
        self.chunk_size = chunk_size
        self.status = run_status or ""
        self.fin_status = fin_status or " " * len(self.status)
        self.unit = unit
        self.seq = sep

    def __get_info(self):
        _info = self.info % (self.title, self.status,
                             self.count/self.chunk_size, self.unit, self.seq, self.total/self.chunk_size, self.unit)
        return _info

    def refresh(self, count=1, status=None):
        self.count += count
        # if status is not None:
        self.status = status or self.status
        end_str = "\r"
        if self.count >= self.total:
            end_str = '\n'
            self.status = status or self.fin_status
        print(self.__get_info(), end=end_str)
