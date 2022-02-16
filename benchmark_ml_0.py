import os
import os.path as osp
#from time import time
import time
from copy import deepcopy
from collections import Counter, defaultdict

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_validate
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
torch.set_default_tensor_type('torch.cuda.FloatTensor')
#torch.manual_seed(0) # for reproduce

import pickle
from grover.util.metrics import get_metric_func
import warnings
warnings.filterwarnings("ignore")

from BBB_score import bbbscore
from argparse import ArgumentParser

def random_sampling(classifier, X_pool, n_instances):
    query_idx = list(np.random.permutation(range(len(X_pool)))[:n_instances])
    if len(query_idx) < n_instances:
        query_idx = list(query_idx * (int(n_instances / len(query_idx)) + 1))[:n_instances]
    return query_idx, X_pool[query_idx]


def data_report(y):
    count = dict(Counter(y))
    ss = sum(dict(Counter(y)).values())
    for key, value in count.items():
        print(f'class {key} has {value} insatances, which make up {100 * round(value/ss, 3):.2f} % total data.')


def featurization(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # AllChem.AddHs(mol)
    ecfp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024))
    return ecfp


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

def load_train_data(validation=False):
    if not validation:
        raw_filename = './dataset/MoleculeNet-BBBP-process-flow-step5-traindata.csv'
        train_data = pd.read_csv(raw_filename)
    else:
        raw_filename = './dataset/moleculenet_bbbp.csv'
        train_data = pd.read_csv(raw_filename)

    feature_filename = raw_filename.replace('.csv', '.pickle')
    filename = raw_filename.replace('.csv', '')

    smiles = train_data.iloc[:, 0].values
    data_x = np.array([featurization(s) for s in smiles])
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
            mol = Chem.MolFromSmiles(smiles)
            canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
        except:
            print('smiles can not convert to mol:' + smiles)
            pass
    train_data_new['cano_smiles'] = canonical_smiles_list
    if os.path.isfile(feature_filename):
        feature_dicts = pickle.load(open(feature_filename, "rb" ))
    else:
        feature_dicts = save_smiles_dicts(smilesList,filename)
    remained_df = train_data_new[train_data_new["cano_smiles"].isin(feature_dicts['smiles_to_atom_mask'].keys())]
    return X, y, remained_df, feature_dicts


def load_test_data():
    raw_filename = "./dataset/Sdata-process-flow-step5-testdata.csv"
    test_data = pd.read_csv(raw_filename)

    feature_filename = raw_filename.replace('.csv', '.pickle')
    filename = raw_filename.replace('.csv', '')

    smiles = test_data.iloc[:, 0].values
    data_x = np.array([featurization(s) for s in smiles])
    data_y = test_data.iloc[:, 1].values
    #shuffle data
    np.random.seed(1234)
    re_indices = np.random.permutation(np.arange(len(test_data)))
    test_data_new = test_data.iloc[re_indices, :]
    X = data_x[re_indices[:]]
    y = data_y[re_indices[:]]

    smilesList = test_data_new.final.values
    canonical_smiles_list = []
    for smiles in smilesList:
        try:
            mol = Chem.MolFromSmiles(smiles)
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
    remained_df['BBB'] = remained_df['max_pooling_label']
    return X, y, remained_df, feature_dicts


def get_estimator(estimator='RF'):
    estimators = {
        'MLP': MLPClassifier(hidden_layer_sizes=(1000, 500),
                             max_iter=3000,
                             batch_size=64,
                             random_state=10,
                             learning_rate_init=0.0001,
                             early_stopping=True
                             ),
        'RF': RandomForestClassifier(n_estimators=100,
                                     max_depth=50,
                                     criterion='entropy',
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
    # shuffle them
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i + batch_size]
        batch_list.append(batch)
    for counter, train_batch in enumerate(batch_list):
        batch_df = dataset.iloc[train_batch, :]
        smiles_list = batch_df.cano_smiles.values

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,
                                                                                                     feature_dicts)
        atoms_prediction, mol_prediction = model(torch.Tensor(x_atom), torch.Tensor(x_bonds),
                                                 torch.cuda.LongTensor(x_atom_index),
                                                 torch.cuda.LongTensor(x_bond_index), torch.Tensor(x_mask))
        model.zero_grad()
        # Step 4. Compute your loss function. (Again, Torch wants the target wrapped in a variable)
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
        # Step 5. Do the backward pass and update the gradient
        #             print(y_val,y_pred,validInds,y_val_adjust,y_pred_adjust)
        loss.backward()
        optimizer.step()


def predict(model, dataset, feature_dicts, loss_function):
    model.eval()
    y_val_list = {}
    y_pred_list = {}
    losses_list = []
    valList = np.arange(0, dataset.shape[0])
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i + batch_size]
        batch_list.append(batch)
    for counter, test_batch in enumerate(batch_list):
        batch_df = dataset.iloc[test_batch, :]
        smiles_list = batch_df.cano_smiles.values

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,
                                                                                                     feature_dicts)
        atoms_prediction, mol_prediction = model(torch.Tensor(x_atom), torch.Tensor(x_bonds),
                                                 torch.cuda.LongTensor(x_atom_index),
                                                 torch.cuda.LongTensor(x_bond_index), torch.Tensor(x_mask))

        atom_pred = atoms_prediction.data[:, :, 1].unsqueeze(2).cpu().numpy()
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

def AttentiveFP_train(model, train_df, valid_df, epochs, feature_dicts, optimizer, loss_function, fold_index, k_index, model_n):
    start_time = str(time.ctime()).replace(':', '-').replace(' ', '_')
    best_param = {}
    best_param["mcc_epoch"] = 0
    best_param["loss_epoch"] = 0
    best_param["valid_mcc"] = 0
    best_param["valid_loss"] = 9e8

    for epoch in range(epochs):
        train_loss, train_y_list, train_y_pred_list = predict(model,
                                                  train_df,
                                                  feature_dicts,
                                                  loss_function)
        valid_loss, valid_y_list, valid_y_pred_list = predict(model,
                                                              valid_df,
                                                              feature_dicts,
                                                              loss_function)
        metric_ls = ['roc_auc', 'prc_auc', 'matthews_corrcoef', 'balanced_accuracy', 'recall', 'precision',
                     'specificity', 'sensitivity']
        # metric_ls = None
        train_score = scoring(train_y_list[0], train_y_pred_list[0], metric_ls)
        valid_score = scoring(valid_y_list[0], valid_y_pred_list[0], metric_ls)

        if valid_score['matthews_corrcoef'] > best_param["valid_mcc"]:
            best_param["mcc_epoch"] = epoch
            best_param["valid_mcc"] = valid_score['matthews_corrcoef']
            if valid_score['matthews_corrcoef'] > 0.55:
                torch.save(model, 'results/AttentiveFP/saved_model/model_' + str(model_n) + '/epochs_' + str(epochs) + '_' + start_time + '_' + str(best_param["mcc_epoch"]) + '.pt')

        if valid_loss < best_param["valid_loss"]:
            best_param["loss_epoch"] = epoch
            best_param["valid_loss"] = valid_loss

        # early stopping
        if (epoch - best_param["mcc_epoch"] > 10) and (epoch - best_param["loss_epoch"] > 12):
            break

        print("EPOCH:\t" + str(epoch))
        print('train_loss:\t' + str(round(train_loss,3)))
        print('train_mcc:\t' + str(round(train_score['matthews_corrcoef'], 3)))
        print('train_ROC:\t' + str(round(train_score['roc_auc'], 3)))
        print('valid_loss:\t' + str(round(valid_loss, 3)))
        for key in train_score.keys():
            print(f'valid_{key}: {valid_score[key]:.4f}')

        train_epoch(model, train_df, feature_dicts, optimizer, loss_function)

    print('mcc_epoch:' + str(best_param["mcc_epoch"]))
    print('loss_epoch:' + str(best_param["loss_epoch"]))
    print('valid_mcc:' + str(best_param["valid_mcc"]))
    print('valid_loss:' + str(best_param["valid_loss"]))

    best_mcc_model = torch.load(
        'results/AttentiveFP/saved_model/model_' + str(model_n) + '/epochs_' + str(
            epochs) + '_' + start_time + '_' + str(best_param["mcc_epoch"]) + '.pt')

    best_loss_model = best_mcc_model
    return best_mcc_model, best_loss_model

#RF; MLP
def run_training(val_data, estimator_name='RF'):
    X_test, y_test, test_data, test_feature_dicts = load_test_data()
    if not val_data:
        X_train, y_train, train_data, train_feature_dicts = load_train_data(validation=False)

        estimator = get_estimator(estimator=estimator_name)
        estimator.fit(X_train, y_train)
        metric_ls = ['roc_auc', 'prc_auc', 'matthews_corrcoef', 'balanced_accuracy', 'recall', 'precision',
                     'specificity', 'sensitivity']
        # metric_ls = None
        test_score = scoring(y_test, estimator.predict_proba(X_test)[:, 1], metric_ls)
        train_score = scoring(y_train, estimator.predict_proba(X_train)[:, 1], metric_ls)

        for key in train_score.keys():
            print(f'test_{key}: {test_score[key]:.4f}')

    else:
        X_train, y_train, train_data, train_feature_dicts = load_train_data(validation=True)
        data_report(y_train)
        data_report(y_test)

        estimator = get_estimator(estimator=estimator_name)
        # kf = KFold(n_splits=3)
        kf = KFold(n_splits=3, shuffle=True, random_state=0)
        train_monitors = defaultdict(list)
        val_monitors = defaultdict(list)
        test_monitors = defaultdict(list)

        for train_index, val_index in kf.split(X_train):
            estimator.fit(X_train[train_index], y_train[train_index])
            test_score = scoring(y_test, estimator.predict_proba(X_test)[:, 1])
            train_score = scoring(y_train[train_index], estimator.predict_proba(X_train[train_index])[:, 1])
            val_score = scoring(y_train[val_index], estimator.predict_proba(X_train[val_index])[:, 1])
            for key in train_score.keys():
                val_monitors[key].append(val_score[key])
                test_monitors[key].append(test_score[key])
                train_monitors[key].append(train_score[key])
        for key in val_monitors.keys():
            for data, monitors in zip(['train', 'val', 'test'], [train_monitors, val_monitors, test_monitors]):
                print(f'{data} {key} = {np.mean(monitors[key]):.3}±{np.std(monitors[key]):.2}')


def run_training_AttentiveFP(val_data,model_n=0):

    X_test, y_test, test_data, test_feature_dicts = load_test_data()
    if not val_data:   #val_data = False
        X_train, y_train, train_data, train_feature_dicts = load_train_data(validation=False)
        train_monitors = defaultdict(list)
        val_monitors = defaultdict(list)
        Stest_monitors = defaultdict(list)

        smilesList = train_data.cano_smiles.values
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(
            [smilesList[0]],
            train_feature_dicts)
        num_atom_features = x_atom.shape[-1]
        num_bond_features = x_bonds.shape[-1]

        loss_function = [nn.CrossEntropyLoss() for i in range(len(tasks))]

        model = Fingerprint(radius, T, num_atom_features, num_bond_features,
                            fingerprint_dim, output_units_num, p_dropout)
        model.cuda()
        optimizer = optim.Adam(model.parameters(), 10 ** -learning_rate, weight_decay=10 ** -weight_decay)
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)
        fold_index = 0
        k_index = 0
        best_mcc_model, best_loss_model = AttentiveFP_train(model, train_data, train_data, epochs, train_feature_dicts,
                                                            optimizer, loss_function, fold_index, k_index, model_n)
        best_model = best_mcc_model
        train_loss, train_y_list, train_y_pred_list = predict(best_model,
                                                              train_data,
                                                              train_feature_dicts,
                                                              loss_function)
        valid_loss, valid_y_list, valid_y_pred_list = predict(best_model,
                                                              train_data,
                                                              train_feature_dicts,
                                                              loss_function)
        Stest_loss, Stest_y_list, Stest_y_pred_list = predict(best_model,
                                                              test_data,
                                                              test_feature_dicts,
                                                              loss_function)

        metric_ls = ['roc_auc', 'prc_auc', 'matthews_corrcoef', 'balanced_accuracy', 'recall', 'precision','specificity', 'sensitivity']
        train_score = scoring(train_y_list[0], train_y_pred_list[0], metric_ls)
        valid_score = scoring(valid_y_list[0], valid_y_pred_list[0], metric_ls)
        Stest_score = scoring(Stest_y_list[0], Stest_y_pred_list[0], metric_ls)
        with open('results/AttentiveFP/result_traindata_best_mcc_64_3_5_epoch_300_model_' + str(model_n) + '.csv', 'w') as ff:
            ff.write('metric,train,valid,Stest' + '\n')
            for key in train_score.keys():
                ff.write(key + ',' + str(train_score[key]) + ',' + str(valid_score[key]) + ',' + str(Stest_score[key]) + '\n')
                train_monitors[key].append(train_score[key])
                #val_monitors[key].append(valid_score[key])
                Stest_monitors[key].append(Stest_score[key])

    else:   #val_data = True
        X_train, y_train, train_data, train_feature_dicts = load_train_data(validation=True)
        data_report(y_train)
        data_report(y_test)
        kf = KFold(n_splits=3, shuffle=True, random_state=0)
    return best_mcc_model



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--benchmark_name', type=str, required=True,
                        help='Run benchmark for BBBp prediction')
    args = parser.parse_args()

    model_name = args.benchmark_name
    if model_name == "BBBscore":
        file_dir = './dataset/Sdata-process-flow-step5-testdata_with_pka.csv'
        df = pd.read_csv(file_dir)
        smi_ls = df['SMILES'].tolist()
        mae_pka_ls = df['maestro_final_pka'].tolist()

        mae_b_new_ls = []
        mae_pred_ls = []
        for i in range(len(smi_ls)):
            mol = Chem.MolFromSmiles(smi_ls[i])
            bbb_score = bbbscore(mol, mae_pka_ls[i])
            mae_b_new = bbb_score[9]
            mae_b_new_ls.append(mae_b_new)
            if mae_b_new >= 4 and mae_b_new <= 6:
                mae_pred_ls.append(1)
            elif mae_b_new < 4 and mae_b_new >= 0:
                mae_pred_ls.append(0)
            else:
                print(smi_ls[i] + ': ' + str(mae_b_new))

        df['BBB_score_with_maestro_pka'] = mae_b_new_ls
        df['BBB_score_pred_label'] = mae_pred_ls
        score = scoring(df['BBB'], mae_b_new_ls, ['roc_auc', 'prc_auc'])
        for key in score.keys():
            print(f'{key} = {np.mean(score[key]):.4}')
        score = scoring(df['BBB'], mae_pred_ls, ['matthews_corrcoef', 'balanced_accuracy', 'recall', 'precision','specificity'])
        for key in score.keys():
            print(f'{key} = {np.mean(score[key]):.4}')

    elif model_name in ['RF', 'MLP']:
        val_data = False  # False:M-data train,predict S-data
        print(model_name + ' model Testing')
        run_training(val_data, estimator_name=model_name)

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
        seeds = [0, 1000, 6000, 9000, 12000]
        loss_function = [nn.CrossEntropyLoss() for i in range(len(tasks))]
        X_test, y_test, test_data, test_feature_dicts = load_test_data()
        Stest_monitors = defaultdict(list)
        y_pred = []
        for i in range(5):
            print('training model_' + str(i))
            torch.manual_seed(seeds[i])
            best_model = run_training_AttentiveFP(val_data, model_n=i)
            Stest_loss, Stest_y_list, Stest_y_pred_list = predict(best_model,
                                                                  test_data,
                                                                  test_feature_dicts,
                                                                  loss_function)
            y_pred.append(Stest_y_pred_list[0])

        y_pred_mean = np.mean(y_pred, axis=0)
        metric_ls = ['roc_auc', 'prc_auc', 'matthews_corrcoef', 'balanced_accuracy', 'recall', 'precision',
                     'specificity', 'sensitivity']
        Stest_score = scoring(Stest_y_list[0], y_pred_mean, metric_ls)
        print('AttentiveFP Ensemble Testing')
        for key in Stest_score.keys():
            print(f'{key} = {np.mean(Stest_score[key]):.4}')


