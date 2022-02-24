import os
import os.path as osp
#from time import time
import time
from copy import deepcopy
from collections import Counter, defaultdict

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from rdkit.Chem import AllChem
from rdkit import Chem

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from AttentiveFP import Fingerprint, Fingerprint_viz, save_smiles_dicts, get_smiles_dicts, get_smiles_array, moltosvg_highlight
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
#torch.manual_seed(0) # for reproduce

import pickle
from grover.util.metrics import get_metric_func
import warnings
warnings.filterwarnings("ignore")

from BBB_score import bbbscore
from argparse import ArgumentParser


def random_sampling(X_pool, n_instances):
    query_idx = list(np.random.permutation(range(len(X_pool)))[:n_instances])
    if len(query_idx) < n_instances:
        query_idx = list(query_idx * (int(n_instances / len(query_idx)) + 1))[:n_instances]
    return query_idx, X_pool[query_idx]


def data_report(y):
    count = dict(Counter(y))
    ss = sum(dict(Counter(y)).values())
    for key, value in count.items():
        print(f'class {key} has {value} insatances, which make up {100 * round(value/ss, 3):.2f} % total data.')

def featurization_ECFP(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # AllChem.AddHs(mol)
    ecfp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024))
    return ecfp

def featurization_PCP(smiles):
    from grover.data.molfeaturegenerator import rdkit_2d_features_generator
    mol = Chem.MolFromSmiles(smiles)
    features = np.array(rdkit_2d_features_generator(mol))  #rdkit_2d_normalized
    return features

def scoring(y, y_pred, metrics=None):
    if not metrics:
        metrics = ['roc_auc', 'prc_auc', 'matthews_corrcoef', 'balanced_accuracy']
    return {m: get_metric_func(m)(y, y_pred) for m in metrics}

def smooth(a, WSZ):
    """
    yy = smooth(y) smooths the data in the column vector y ..
    The first few elements of yy are given by
    yy(1) = y(1)
    yy(2) = (y(1) + y(2) + y(3))/3
    yy(3) = (y(1) + y(2) + y(3) + y(4) + y(5))/5
    yy(4) = (y(2) + y(3) + y(4) + y(5) + y(6))/5
    :param a: NumPy 1-D array containing the data to be smoothed
    :param WSZ: moothing window size needs, which must be odd number
    """
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid')/WSZ
    r = np.arange(1, WSZ-1, 2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((start, out0, stop))

def average_lineplot(monitor_dict_1, monitor_dict_2, monitor_dict_3, step=5, save_dir=None):
    if save_dir: os.makedirs(save_dir, exist_ok=True)
    for key in monitor_dict_1.keys():
        score1 = monitor_dict_1[key]
        score2 = monitor_dict_2[key]
        if monitor_dict_3:
            score3 = monitor_dict_3[key]

        moving_avg1 = smooth(score1, step)
        moving_avg2 = smooth(score2, step)
        if monitor_dict_3:
            moving_avg3 = smooth(score3, step)
        ax = plt.figure()
        ax = sns.lineplot(x=range(len(moving_avg1)), y=moving_avg1, label='random sampling')
        ax = sns.lineplot(x=range(len(moving_avg1)), y=moving_avg2, label='margin sampling')
        if monitor_dict_3:
            ax = sns.lineplot(x=range(len(moving_avg3)), y=moving_avg3, label='entropy sampling')
        if key == 'matthews_corrcoef':
            ax.set_ylim(0.3, 0.8)
        else:
            ax.set_ylim(0.5, 1)
        ax.set_xlabel('Queries')
        ax.set_ylabel(key.replace('_', ' ').title())
        ax.set_title(f'MOVING AVERAGE SCORE (WINDOWS = {step})')
        if save_dir:
            plt.savefig(osp.join(save_dir, f'{key}.png'))
    plt.close('all')

def load_train_data_for_uncertainty(raw_filename, prop='ECFP'):

    train_data = pd.read_csv(raw_filename)
    feature_filename = raw_filename.replace('.csv', '.pickle')
    filename = raw_filename.replace('.csv', '')
    scaler = None
    smiles = train_data.iloc[:, 0].values
    if prop == 'ECFP':
        data_x = np.array([featurization_ECFP(s) for s in smiles])
    elif prop == 'PCP':
        data_x = np.array([featurization_PCP(s) for s in smiles])
        mm = MinMaxScaler()
        scaler = mm.fit(data_x)
        data_x = scaler.transform(data_x)
    data_y = train_data.iloc[:, 1].values
    #shuffle data
    np.random.seed(1234)
    re_indices = np.random.permutation(np.arange(len(train_data)))
    train_data_new = train_data.iloc[re_indices, :]
    X = data_x[re_indices[:]]
    y = data_y[re_indices[:]]

    smilesList = train_data_new.SMILES.values
    canonical_smiles_list = []
    for smiles in smilesList:
        try:
            canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
        except:
            print('smiles can not convert to mol:' + smiles)
            pass
    train_data_new['cano_smiles'] = canonical_smiles_list
    if os.path.isfile(feature_filename):
        feature_dicts = pickle.load(open(feature_filename, "rb"))
    else:
        feature_dicts = save_smiles_dicts(smilesList, filename)
    remained_df = train_data_new[train_data_new["cano_smiles"].isin(feature_dicts['smiles_to_atom_mask'].keys())]
    return X, y, remained_df, feature_dicts, scaler, train_data_new

def load_test_data_for_uncertainty(raw_filename, prop='ECFP', scaler=None):
    test_data = pd.read_csv(raw_filename)
    feature_filename = raw_filename.replace('.csv', '.pickle')
    filename = raw_filename.replace('.csv', '')

    smiles = test_data.iloc[:, 0].values
    if prop == 'ECFP':
        data_x = np.array([featurization_ECFP(s) for s in smiles])
    elif prop == 'PCP':
        data_x = np.array([featurization_PCP(s) for s in smiles])
        if scaler != None:
            data_x = scaler.transform(data_x)
    data_y = test_data.iloc[:, 1].values
    #shuffle data
    np.random.seed(1234)
    re_indices = np.random.permutation(np.arange(len(test_data)))
    test_data_new = test_data.iloc[re_indices, :]
    X = data_x[re_indices[:]]
    y = data_y[re_indices[:]]

    smilesList = test_data_new.SMILES.values
    canonical_smiles_list = []
    for smiles in smilesList:
        try:
            canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
        except:
            print('smiles can not convert to mol:' + smiles)
            pass
    test_data_new['cano_smiles'] = canonical_smiles_list
    if os.path.isfile(feature_filename):
        feature_dicts = pickle.load(open(feature_filename, "rb"))
    else:
        feature_dicts = save_smiles_dicts(smilesList, filename)
    remained_df = test_data_new[test_data_new["cano_smiles"].isin(feature_dicts['smiles_to_atom_mask'].keys())]
    remained_df['BBB'] = remained_df['BBB']
    return X, y, remained_df, feature_dicts, test_data_new

def load_train_data(validation=False, prop='ECFP'):
    if not validation:
        raw_filename = './dataset/MoleculeNet-BBBP-process-flow-step5-traindata.csv'
        train_data = pd.read_csv(raw_filename)
    else:
        raw_filename = './dataset/moleculenet_bbbp.csv'
        train_data = pd.read_csv(raw_filename)

    feature_filename = raw_filename.replace('.csv', '.pickle')
    filename = raw_filename.replace('.csv', '')
    scaler = None
    smiles = train_data.iloc[:, 0].values
    if prop == 'ECFP':
        data_x = np.array([featurization_ECFP(s) for s in smiles])
    elif prop == 'PCP':
        data_x = np.array([featurization_PCP(s) for s in smiles])
        mm = MinMaxScaler()
        scaler = mm.fit(data_x)
        data_x = scaler.transform(data_x)
    data_y = train_data.iloc[:, 1].values
    #shuffle data
    np.random.seed(1234)
    re_indices = np.random.permutation(np.arange(len(train_data)))
    train_data_new = train_data.iloc[re_indices, :]
    X = data_x[re_indices[:]]
    y = data_y[re_indices[:]]

    smilesList = train_data_new.SMILES.values
    canonical_smiles_list = []
    for smiles in smilesList:
        try:
            canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
        except:
            print('smiles can not convert to mol:' + smiles)
            pass
    train_data_new['cano_smiles'] = canonical_smiles_list
    if os.path.isfile(feature_filename):
        feature_dicts = pickle.load(open(feature_filename, "rb"))
    else:
        feature_dicts = save_smiles_dicts(smilesList, filename)
    remained_df = train_data_new[train_data_new["cano_smiles"].isin(feature_dicts['smiles_to_atom_mask'].keys())]
    return X, y, remained_df, feature_dicts, scaler, train_data_new

def load_test_data(prop='ECFP', scaler=None):
    raw_filename = "./dataset/Sdata-process-flow-step5-testdata.csv"
    test_data = pd.read_csv(raw_filename)

    feature_filename = raw_filename.replace('.csv', '.pickle')
    filename = raw_filename.replace('.csv', '')

    smiles = test_data.iloc[:, 0].values
    if prop == 'ECFP':
        data_x = np.array([featurization_ECFP(s) for s in smiles])
    elif prop == 'PCP':
        data_x = np.array([featurization_PCP(s) for s in smiles])
        if scaler != None:
            data_x = scaler.transform(data_x)
    data_y = test_data.iloc[:, 1].values
    #shuffle data
    np.random.seed(1234)
    re_indices = np.random.permutation(np.arange(len(test_data)))
    test_data_new = test_data.iloc[re_indices, :]
    X = data_x[re_indices[:]]
    y = data_y[re_indices[:]]

    smilesList = test_data_new.SMILES.values
    canonical_smiles_list = []
    for smiles in smilesList:
        try:
            canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
        except:
            print('smiles can not convert to mol:' + smiles)
            pass
    test_data_new['cano_smiles'] = canonical_smiles_list
    if os.path.isfile(feature_filename):
        feature_dicts = pickle.load(open(feature_filename, "rb"))
    else:
        feature_dicts = save_smiles_dicts(smilesList, filename)
    remained_df = test_data_new[test_data_new["cano_smiles"].isin(feature_dicts['smiles_to_atom_mask'].keys())]
    remained_df['BBB'] = remained_df['BBB']
    return X, y, remained_df, feature_dicts, test_data_new

def get_estimator(estimator='MLP(PCP)'):
    estimators = {
        'MLP(ECFP)': MLPClassifier(hidden_layer_sizes=(1000, 500),
                             max_iter=3000,
                             batch_size=64,
                             random_state=10,
                             learning_rate_init=0.0001,
                             early_stopping=True
                             ),

        'RF(ECFP)': RandomForestClassifier(n_estimators=100,
                                     max_depth=50,
                                     criterion='entropy',
                                     n_jobs=-1,
                                     random_state=10,
                                     oob_score=True
                                     ),

        'MLP(PCP)': MLPClassifier(hidden_layer_sizes=(1500, 1000, 500),
                                 max_iter=1000,
                                 batch_size=16,
                                 random_state=10,
                                 learning_rate_init=0.0001,
                                 early_stopping=True
                                 ),

        'RF(PCP)': RandomForestClassifier(n_estimators=250,
                                     max_depth=20,
                                     criterion='gini',
                                     n_jobs=-1,
                                     random_state=10,
                                     oob_score=True
                                     )
    }
    return estimators[estimator]


#AttentiveFP
def train_epoch(model, dataset, feature_dicts, optimizer, loss_function):
    ###Trains a model for an epoch.
    model.train()
    valList = np.arange(0, dataset.shape[0])
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i + batch_size]
        batch_list.append(batch)
    for counter, train_batch in enumerate(batch_list):
        batch_df = dataset.iloc[train_batch, :]
        smiles_list = batch_df.cano_smiles.values

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,
                                                                                                     feature_dicts)

        atoms_prediction, mol_prediction, _ = model(torch.Tensor(x_atom).to('cuda'),
                                                 torch.Tensor(x_bonds).to('cuda'),
                                                 torch.cuda.LongTensor(x_atom_index),
                                                 torch.cuda.LongTensor(x_bond_index),
                                                 torch.Tensor(x_mask).to('cuda')
                                                 )
        model.zero_grad()

        loss = 0.0
        for i, task in enumerate(tasks):
            y_pred = mol_prediction[:, i * per_task_output_units_num:(i + 1) *
                                                                     per_task_output_units_num]
            y_val = batch_df[task].values

            validInds = np.where((y_val == 0) | (y_val == 1))[0]
            if len(validInds) == 0:
                continue
            y_val_adjust = np.array([y_val[v] for v in validInds]).astype(float)
            validInds = torch.cuda.LongTensor(validInds).squeeze()
            y_pred_adjust = torch.index_select(y_pred, 0, validInds)

            loss += loss_function[i](
                y_pred_adjust,
                torch.cuda.LongTensor(y_val_adjust))

        loss.backward()
        optimizer.step()

def predict(model, dataset, feature_dicts, loss_function):
    model.eval()
    y_val_list = {}
    y_pred_list = {}
    losses_list = []
    valList = np.arange(0, dataset.shape[0])
    batch_list = []
    batch_size = 64
    tasks = ['BBB']
    per_task_output_units_num = 2

    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i + batch_size]
        batch_list.append(batch)
    for counter, test_batch in enumerate(batch_list):
        batch_df = dataset.iloc[test_batch, :]
        smiles_list = batch_df.cano_smiles.values

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,
                                                                                                     feature_dicts)

        atoms_prediction, mol_prediction, _ = model(torch.Tensor(x_atom).to('cuda'),
                                                 torch.Tensor(x_bonds).to('cuda'),
                                                 torch.cuda.LongTensor(x_atom_index),
                                                 torch.cuda.LongTensor(x_bond_index),
                                                 torch.Tensor(x_mask).to('cuda')
                                                 )

        for i, task in enumerate(tasks):
            y_pred = mol_prediction[:, i * per_task_output_units_num:(i + 1) *
                                                                     per_task_output_units_num]
            y_val = batch_df[task].values

            validInds = np.where((y_val == 0) | (y_val == 1))[0]
            if len(validInds) == 0:
                continue
            y_val_adjust = np.array([y_val[v] for v in validInds]).astype(float)
            validInds = torch.cuda.LongTensor(validInds).squeeze()
            y_pred_adjust = torch.index_select(y_pred, 0, validInds)
            loss = loss_function[i](
                y_pred_adjust,
                torch.cuda.LongTensor(y_val_adjust))
            y_pred_adjust = F.softmax(y_pred_adjust, dim=-1).data.cpu().numpy()[:, 1]
            losses_list.append(loss.cpu().detach().numpy())
            try:
                y_val_list[i].extend(y_val_adjust)
                y_pred_list[i].extend(y_pred_adjust)
            except:
                y_val_list[i] = []
                y_pred_list[i] = []
                y_val_list[i].extend(y_val_adjust)
                y_pred_list[i].extend(y_pred_adjust)
    test_loss = np.array(losses_list).mean()

    return test_loss, y_val_list, y_pred_list

def apply_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.train()

def predict_Ttimes(model, dataset, feature_dicts, loss_function, T=30):

    model.eval()
    model.apply(apply_dropout)

    valList = np.arange(0, dataset.shape[0])
    batch_size = 64
    tasks = ['BBB']
    per_task_output_units_num = 2

    # predict stochastic dropout model T times
    preds_times = []
    for t in range(T):
        if t % 10 == 0: print(f'Have predicted for {t+1}/{T} times')

        y_val_list = {}
        y_pred_list = {}
        losses_list = []
        batch_list = []

        for i in range(0, dataset.shape[0], batch_size):
            batch = valList[i:i + batch_size]
            batch_list.append(batch)
        for counter, test_batch in enumerate(batch_list):
            batch_df = dataset.iloc[test_batch, :]
            smiles_list = batch_df.cano_smiles.values

            x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,
                                                                                                         feature_dicts)
            atoms_prediction, mol_prediction, _ = model(torch.Tensor(x_atom).to('cuda'),
                                                     torch.Tensor(x_bonds).to('cuda'),
                                                     torch.cuda.LongTensor(x_atom_index),
                                                     torch.cuda.LongTensor(x_bond_index),
                                                     torch.Tensor(x_mask).to('cuda')
                                                     )

            for i, task in enumerate(tasks):
                y_pred = mol_prediction[:, i * per_task_output_units_num:(i + 1) *
                                                                         per_task_output_units_num]
                y_val = batch_df[task].values

                validInds = np.where((y_val == 0) | (y_val == 1))[0]
                if len(validInds) == 0:
                    continue
                y_val_adjust = np.array([y_val[v] for v in validInds]).astype(float)
                validInds = torch.cuda.LongTensor(validInds).squeeze()
                y_pred_adjust = torch.index_select(y_pred, 0, validInds)
                loss = loss_function[i](
                    y_pred_adjust,
                    torch.cuda.LongTensor(y_val_adjust))
                y_pred_adjust = F.softmax(y_pred_adjust, dim=-1).data.cpu().numpy()[:, 1]
                losses_list.append(loss.cpu().detach().numpy())
                try:
                    y_val_list[i].extend(y_val_adjust)
                    y_pred_list[i].extend(y_pred_adjust)
                except:
                    y_val_list[i] = []
                    y_pred_list[i] = []
                    y_val_list[i].extend(y_val_adjust)
                    y_pred_list[i].extend(y_pred_adjust)
        preds = y_pred_list[0]
        preds_times.append([p for p in preds])

    p_hat = np.array(preds_times)
    p_hat_binary = np.array([[[1 - p, p] for p in sample] for sample in p_hat])
    return p_hat_binary

def AttentiveFP_train(model, train_df, valid_df, epochs, feature_dicts, optimizer, loss_function, model_n):
    start_time = str(time.ctime()).replace(':', '-').replace(' ', '_')
    best_param = {}
    best_param["mcc_epoch"] = 0
    best_param["loss_epoch"] = 0
    best_param["valid_mcc"] = 0
    best_param["valid_loss"] = 9e8

    save_dir = 'BBBp_results/AttentiveFP'
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)

    for epoch in range(epochs):
        train_loss, train_y_list, train_y_pred_list = predict(model, train_df, feature_dicts, loss_function)
        valid_loss, valid_y_list, valid_y_pred_list = predict(model, valid_df, feature_dicts, loss_function)
        train_score = scoring(train_y_list[0], train_y_pred_list[0])
        valid_score = scoring(valid_y_list[0], valid_y_pred_list[0])

        if valid_score['matthews_corrcoef'] > best_param["valid_mcc"]:
            best_param["mcc_epoch"] = epoch
            best_param["valid_mcc"] = valid_score['matthews_corrcoef']
            if valid_score['matthews_corrcoef'] > 0.55:
                torch.save(model, os.path.join(save_dir, 'model_' + str(model_n), 'epochs_' + str(epochs) + '_' + start_time + '_' + str(best_param["mcc_epoch"]) + '.pt'))

        if valid_loss < best_param["valid_loss"]:
            best_param["loss_epoch"] = epoch
            best_param["valid_loss"] = valid_loss

        # early stopping
        if (epoch - best_param["mcc_epoch"] > 10) and (epoch - best_param["loss_epoch"] > 12):
            break

        print("EPOCH:\t" + str(epoch))
        for key in train_score.keys():
            print(f'train_{key}: {train_score[key]:.4f}')

        train_epoch(model, train_df, feature_dicts, optimizer, loss_function)

    print('mcc_epoch:' + str(best_param["mcc_epoch"]))
    print('loss_epoch:' + str(best_param["loss_epoch"]))
    print('valid_mcc:' + str(best_param["valid_mcc"]))
    print('valid_loss:' + str(best_param["valid_loss"]))

    best_mcc_model = torch.load(
        'BBBp_results/AttentiveFP/model_' + str(model_n) + '/epochs_' + str(
            epochs) + '_' + start_time + '_' + str(best_param["mcc_epoch"]) + '.pt')

    torch.save(best_mcc_model, os.path.join(save_dir, 'model_' + str(model_n), 'model.pt'))

    return best_mcc_model

#RF(ECFP); MLP(ECFP)
def run_training(val_data, estimator_name='RF'):
    X_test, y_test, test_data, test_feature_dicts, test_data_new = load_test_data(prop='ECFP')
    if not val_data:
        X_train, y_train, train_data, train_feature_dicts, scaler, train_data_new = load_train_data(validation=False, prop='ECFP')

        estimator = get_estimator(estimator=estimator_name)
        estimator.fit(X_train, y_train)

        test_score = scoring(y_test, estimator.predict_proba(X_test)[:, 1])
        train_score = scoring(y_train, estimator.predict_proba(X_train)[:, 1])

        save_dir = os.path.join('BBBp_results', estimator_name, 'model_0')
        if os.path.exists(save_dir) == False:
            os.mkdir(save_dir)
        with open(os.path.join(save_dir, 'model.ckpt'), 'wb') as f:
            pickle.dump(estimator, f)

        for key in train_score.keys():
            print(f'train_{key}: {train_score[key]:.4f}')
        for key in train_score.keys():
            print(f'test_{key}: {test_score[key]:.4f}')

        y_pred = estimator.predict_proba(X_test)[:, 1]
        test_data_new['probability'] = y_pred

#RF(PCP); MLP(PCP)
def run_training_PCP(val_data, estimator_name='MLP(PCP)'):
    if not val_data:
        X_train, y_train, train_data, train_feature_dicts, scaler, train_data_new = load_train_data(validation=False, prop='PCP')

        X_test, y_test, test_data, test_feature_dicts, test_data_new = load_test_data(prop='PCP', scaler=scaler)

        estimator = get_estimator(estimator=estimator_name)
        estimator.fit(X_train, y_train)
        test_score = scoring(y_test, estimator.predict_proba(X_test)[:, 1])
        train_score = scoring(y_train, estimator.predict_proba(X_train)[:, 1])

        save_dir = os.path.join('BBBp_results', estimator_name, 'model_0')
        if os.path.exists(save_dir) == False:
            os.mkdir(save_dir)
        with open(os.path.join(save_dir, 'model.ckpt'), 'wb') as f:
            pickle.dump(estimator, f)

        y_pred = estimator.predict_proba(X_test)[:, 1]
        test_data_new['probability'] = y_pred

        for key in train_score.keys():
            print(f'train_{key}: {train_score[key]:.4f}')
        for key in train_score.keys():
            print(f'test_{key}: {test_score[key]:.4f}')


def run_training_AttentiveFP(val_data, model_n=0):
    X_test, y_test, test_data, test_feature_dicts, test_data_new = load_test_data(prop='ECFP', scaler=None)
    if not val_data:
        X_train, y_train, train_data, train_feature_dicts, scaler, train_data_new = load_train_data(validation=False, prop='ECFP')
        smilesList = train_data.cano_smiles.values
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array([smilesList[0]],train_feature_dicts)
        num_atom_features = x_atom.shape[-1]
        num_bond_features = x_bonds.shape[-1]

        loss_function = [nn.CrossEntropyLoss() for i in range(len(tasks))]

        model = Fingerprint(radius, T, num_atom_features, num_bond_features, fingerprint_dim, output_units_num, p_dropout)
        model.cuda()
        optimizer = optim.Adam(model.parameters(), 10 ** -learning_rate, weight_decay=10 ** -weight_decay)

        best_model = AttentiveFP_train(model, train_data, train_data, epochs, train_feature_dicts, optimizer, loss_function, model_n)

        Stest_loss, Stest_y_list, Stest_y_pred_list = predict(best_model,
                                                              test_data,
                                                              test_feature_dicts,
                                                              loss_function)

        Stest_score = scoring(Stest_y_list[0], Stest_y_pred_list[0])

        for key in Stest_score.keys():
            print(f'test_{key} {Stest_score[key]:.4}')




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--benchmark_name', type=str, required=True,
                        help='Run benchmark for BBBp prediction')
    args = parser.parse_args()

    model_name = args.benchmark_name
    if model_name == 'BBBscore':
        file_dir = 'BBBp_results/BBB score/test_result.csv'
        df = pd.read_csv(file_dir)
        smi_ls = df['SMILES'].tolist()
        mae_pka_ls = df['pka'].tolist()
        pred_ls = []
        pred_label_ls = []
        for i in range(len(smi_ls)):
            mol = Chem.MolFromSmiles(smi_ls[i])
            bbb_score = bbbscore(mol, mae_pka_ls[i])
            pred = bbb_score[9]
            pred_ls.append(pred)
            if pred >= 4 and pred <= 6:
                pred_label_ls.append(1)
            elif pred < 4 and pred >= 0:
                pred_label_ls.append(0)
        df['Prediction'] = pred_ls
        df['Prediction_label'] = pred_label_ls
        test_score = scoring(df['Target'], df['Prediction'], ['roc_auc', 'prc_auc'])
        test_score.update(
            scoring(df['Target'], df['Prediction_label'], ['matthews_corrcoef', 'balanced_accuracy']))
        for key in test_score.keys():
            print(f'{key} {test_score[key]:.4}')
        df.to_csv('BBBp_results/BBB score/test_result.csv', index=False)

    elif model_name in ['RF(ECFP)', 'MLP(ECFP)']:
        val_data = False  # False:M-data train,predict S-data
        print(model_name + ' model Testing')
        run_training(val_data, estimator_name=model_name)

    elif model_name in ['RF(PCP)', 'MLP(PCP)']:
        val_data = False  # False:M-data train,predict S-data
        print(model_name + ' model with physicochemical property Testing')
        run_training_PCP(val_data, estimator_name=model_name)

    elif model_name == 'AttentiveFP':
        val_data = False
        tasks = ['BBB']
        per_task_output_units_num = 2
        output_units_num = len(tasks) * per_task_output_units_num
        batch_size = 64
        epochs = 300
        p_dropout = 0.1
        fingerprint_dim = 150
        radius = 3
        T = 2
        weight_decay = 2.9  # also known as l2_regularization_lambda
        learning_rate = 3.5  # 3.5
        seeds = 0
        loss_function = [nn.CrossEntropyLoss() for i in range(len(tasks))]
        torch.manual_seed(seeds)
        run_training_AttentiveFP(val_data, model_n=0)




