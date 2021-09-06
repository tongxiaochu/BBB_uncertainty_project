"""
Uncertainty measures and uncertainty based sampling strategies for the active learning models.
"""


import numpy as np
from scipy.stats import entropy
from sklearn import preprocessing

from utils.utils import get_fps, tanimoto_distance, featurize_loader, standardized_euclidean_distance


def mc_dropout(preds_probas):
    '''https://www.depends-on-the-definition.com/model-uncertainty-in-deep-learning-with-monte-carlo-dropout/'''
    posterior_vars = np.std(preds_probas, axis=0) ** 2
    posterior_vars_c0 = posterior_vars[:, 0]
    return posterior_vars_c0


def multi_initial(preds_probas):
    return np.var(preds_probas, axis=0).ravel()


def entropy_uncertainty(preds_proba) -> np.ndarray:
    """
    Entropy of predictions of the for the provided samples.

    Returns:
        Entropy of the class probabilities.
    """
    return np.array(entropy(np.transpose(preds_proba)))


def fingerprint_distance(smiles_train, smiles_val, fp):
    """
    Compute uncertainty using molecule fingerprint distance
    """
    sup_fps = get_fps(smiles_train, fp)
    qu_fps = get_fps(smiles_val, fp)
    ta_dist = tanimoto_distance(sup_fps, qu_fps)
    min_tad = [min(query_mol_dist) for query_mol_dist in ta_dist]
    return np.array(min_tad)


def latent_distance(network,
                    X_train,
                    X_val,
                    args
                    ):
    """
    Compute uncertainty using latent space distance
    """
    train_features = featurize_loader(network, X_train, args)
    train_features_scale = preprocessing.scale(train_features, axis=0)

    val_features = featurize_loader(network, X_val, args)
    val_features_scale = preprocessing.scale(val_features, axis=0)

    eu_dist = standardized_euclidean_distance(train_features_scale, val_features_scale)
    min_eud = [min(query_mol_dist) for query_mol_dist in eu_dist]
    return np.array(min_eud)


