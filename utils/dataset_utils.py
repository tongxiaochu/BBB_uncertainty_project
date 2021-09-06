from typing import Tuple, Dict, List, Union, Any
from argparse import Namespace
from logging import Logger
from collections import OrderedDict
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from grover.util.metrics import get_metric_func
from grover.data import MoleculeDataset, MolGraph, BatchMolGraph
from grover.model.models import GroverFinetuneTask

Model = Union[GroverFinetuneTask]


def _init_fn():
    np.random.seed(123)


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


class FeatureCollator(object):
    """
    Collator for pytorch dataloader
    """

    def __call__(self, batch):
        smiles_batch = [d.smiles for d in batch]
        features_batch = [d.features for d in batch]
        target_batch = [d.targets for d in batch]
        mask = [[x is not None for x in tb] for tb in target_batch]
        targets = [[0 if x is None else x for x in tb] for tb in target_batch]
        return smiles_batch, features_batch, targets, mask


def load_MoleculeDataset(data: MoleculeDataset, args: Namespace) -> Tuple[np.array, np.array, np.array, np.array]:
    mol_collator = FeatureCollator()
    loader = DataLoader(
        data,
        batch_size=len(data),
        shuffle=False,
        num_workers=0,
        collate_fn=mol_collator,
        worker_init_fn=_init_fn
    )
    smiles, features, y_train, mask = next(iter(loader))

    mol_features = []
    for s, f in zip(smiles, features):
        mol_graph = MolGraph(s, args)
        mol_features.append([mol_graph, f])
    smiles = np.array(smiles)
    mol_features = np.array(mol_features, dtype=object)
    mask = np.array(mask)
    y_train = np.array(y_train)

    return smiles, mol_features, y_train, mask


class MyDataset(Dataset):
    def __init__(self, x: np.array, y: np.array, mask: np.array):
        self.x = x
        self.y = y
        self.mask = mask

    def __getitem__(self, index: int) -> Tuple[np.array, np.array, np.array]:
        return self.x[index], self.y[index], self.mask[index]

    def __len__(self) -> int:
        return len(self.x)


class GraphCollator:
    """
    Collator for pytorch dataloader
    :param args: Arguments.
    """
    def __init__(self,  args: Namespace):
        self.args = args

    def __call__(self, batch: Tuple[np.array, np.array, np.array]) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        graphs = [d[0][0] for d in batch]
        graph_batch = BatchMolGraph(graphs, self.args)
        graph_feature_batch = graph_batch.get_components()
        mol_features_batch = [d[0][1] for d in batch]
        y_batch = torch.Tensor([d[1] for d in batch])
        mask_batch = torch.Tensor([d[2] for d in batch])
        return graph_feature_batch, mol_features_batch, y_batch, mask_batch


