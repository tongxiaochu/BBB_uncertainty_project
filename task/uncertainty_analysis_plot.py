# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:26:09 2020

@author: amberxtli
"""
import os
import numpy as np
import pandas as pd

from argparse import ArgumentParser

from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
from grover.util.metrics import get_metric_func
import seaborn as sns
import matplotlib.pyplot as plt


def calc_MODZ(array):
    if len(array) == 2:
        return np.mean(array, 0)
    else:
        CM = spearmanr(array.T)[0]
        fil = CM < 0
        CM[fil] = 0.01
        weights = np.sum(CM, 1) - 1
        weights = weights / np.sum(weights)
        weights = weights.reshape((-1, 1))
        return np.dot(array.T, weights).reshape((-1, 1)[0])


def normalize_uncertainty(result_df):
    np.random.seed(123)
    random_uncertainty = np.random.rand(result_df.shape[0])
    result_df['Random'] = random_uncertainty
    for uc in ['FPsDist', 'LatentDist', 'Entropy', 'MC-dropout', 'Multi-initial']:
        result_df[uc] = MinMaxScaler().fit_transform(result_df[uc].values.reshape(-1, 1))
    return result_df


def ensemble_uncertainty(result_df, ensemble_unames=['Entropy', 'MC-dropout']):
    ensemble_u = calc_MODZ(result_df[ensemble_unames].values.T)
    result_df['Ensemble'] = ensemble_u
    return result_df


def cal_scores(result_df, uncertainties, uncertainty_metrics):
    wds = 0.1
    dataframe_list = []
    for i in range(len(uncertainty_metrics)):
        uc, uncertainty = uncertainties[i], uncertainty_metrics[i]
        percentile_index = np.concatenate((len(uc) * np.arange(0, 1, wds), [len(uc) - 1]))
        percentile_index = [np.argsort(uc)[int(p)] for p in percentile_index]
        pc = percentile_cutoffs = [uc[p] for p in percentile_index]
        for j in range(len(percentile_cutoffs) - 1):
            sub_df = result_df[(uc <= pc[::-1][j]) & (uc > pc[::-1][-1])]
            y_pred = sub_df[['Prediction']].values
            y = sub_df[['Target']].values
            mcc_score = get_metric_func('matthews_corrcoef')(y, y_pred)
            dataframe_list.append([uncertainty, 'MCC', j * wds, mcc_score])
    scores_df = pd.DataFrame(dataframe_list, columns=['uncertainty', 'metrics', 'percentile', 'performance'])
    return scores_df


def draw_lineplot(result_df, uname, args):
    scores_df = cal_scores(result_df, result_df[uname].values.T, uname)

    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    sns.set_style("whitegrid", {"ytick.major.size": 0.1,
                                "ytick.minor.size": 0.1,
                                "grid.linestyle": '--'}
                  )
    scores_df[scores_df['performance'] == 0] = np.nan
    sns.lineplot(data=scores_df, x='percentile', y='performance', hue='uncertainty',
                 hue_order=uname, ax=axes, palette=[colors[um] for um in uname],  legend=True)
    sns.scatterplot(data=scores_df, x='percentile', y='performance', hue='uncertainty',
                 marker='o', s=120, ax=axes, palette=[colors[um] for um in uname],  legend=False)
    axes.set_xlim([-0.05, 1])

    axes.set_ylim([0.2, 1.05])

    # axes.set_title('Attentive FP', size=22)
    axes.set_ylabel('MCC')
    for l in axes.lines:
        l.set_lw(3)
    axes.set_xticks(np.arange(0, 1, 0.1))
    axes.set_xticklabels([f'{i:.1f}' for i in np.arange(0.1, 1.1, 0.1)[::-1]])
    axes.set_ylabel(axes.get_ylabel(), size=22)
    axes.set_xlabel('percentile', size=22)
    axes.tick_params(labelsize=20)
    axes.grid(which='minor', axis='x', linestyle='--', linewidth=0.55)
    plt.legend(fontsize='22')
    plt.subplots_adjust(wspace=0.25)
    plt.savefig(os.path.join(args.save_dir, 'uncertainty_lineplot_'+'_'.join(uname)+'.png'), dpi=200, bbox_inches='tight')


def draw_histplot(result_df, uname, args):
    if len(uname) > 1:
        fig, axes = plt.subplots(2, int(len(uname)/2), figsize=(6*len(uname)/2, 10))
    elif len(uname) == 1:
        fig, axes = plt.subplots(1, 1, figsize=(5, 5))
        axes = np.array([axes])
    sns.set_style("whitegrid", {"xtick.major.size": 0.1,
                                "xtick.minor.size": 0.05,
                                "grid.linestyle": '--'}
                  )

    palette = sns.color_palette(['steelblue', 'lightskyblue', 'crimson', 'coral'])
    for i in range(len(uname)):
        ax = axes.ravel()[i]
        sns.histplot(data=result_df, x=uname[i], binrange=(0, 1), binwidth=0.1, hue='Confuse', stat='density',
                     multiple="fill", fill=True, ax=ax, palette=palette, hue_order=['TP', 'TN', 'FP', 'FN'], legend=i==0 )
        ax.set_xlabel(uname[i], size=18)
        if i in (0, 3):
            ax.set_ylabel('percentage', size=18)
        else:
            ax.set_ylabel('')
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.set_yticklabels([f'{i:.0%}' for i in np.arange(0, 1.1, 0.2)])
        ax.tick_params(labelsize=16)

        ax2 = ax.twinx()
        sns.histplot(data=result_df, x=uname[i], binrange=(0, 1), binwidth=0.1, stat='count',
                     element="poly", fill=False, ax=ax2, color=(0, 220 / 255, 220 / 255))
        if i in (2, 5):
            ax2.set_ylabel('count of samples', size=18)
        else:
            ax2.set_ylabel('')
        ax2.tick_params(labelsize=16)
        ax2.set_xticks(np.arange(0, 1.1, 0.2))
        ax2.set_xlim((0, 1))
        ax2.set_ylim((0))
        for l in ax2.get_lines():
            l.set_lw(1.5)
            l.set_marker('o')
            l.set_linestyle('--')
        if i == 0:
            plt.setp(ax.get_legend().get_texts(), fontsize='12')
            ax.get_legend().set_title('')
            ax.get_legend()._set_loc((0.75, 0.7))
    plt.subplots_adjust(wspace=0.35, hspace=0.25)
    plt.savefig(os.path.join(args.save_dir, 'uncertainty_histplot_'+'_'.join(uname)+'.png'), dpi=200, bbox_inches='tight')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to result CSV')
    args = parser.parse_args()

    print("Load '{}' and draw figures.".format(args.save_dir))
    result_df = pd.read_csv(os.path.join(args.save_dir, f'test_result.csv'))

    result_df = ensemble_uncertainty(normalize_uncertainty(result_df), ensemble_unames=['Entropy', 'MC-dropout'])

    colors = {k: v for v, k in zip([c for i, c in enumerate(sns.color_palette("RdBu_r", 9)) if i not in (3, 4, 5)],
                                   ['Entropy', 'Multi-initial', 'FPsDist', 'LatentDist', 'MC-dropout', 'Ensemble'])}
    colors['Random'] = (0.5, 0.5, 0.5)
    args.colors = colors

    # figure 3
    model_dir = args.save_dir.split('/')[-1]

    if 'mlp' in model_dir:
        uname = ['Entropy', 'Multi-initial', 'FPsDist', 'LatentDist', 'Random']
    elif 'rf' in model_dir:
        uname = ['Entropy', 'Multi-initial', 'FPsDist', 'Random']
    elif 'AttentiveFP' in model_dir:
        uname = ['Entropy', 'Multi-initial', 'FPsDist', 'LatentDist', 'MC-dropout', 'Random']
    else:
        uname = ['Entropy', 'Multi-initial', 'FPsDist', 'LatentDist', 'MC-dropout', 'Random']

    print("Draw Figure 3 saved in {}".format(
        os.path.join(args.save_dir, 'uncertainty_lineplot_' + '_'.join(uname) + '.png')))

    draw_lineplot(result_df, uname=uname, args=args)

    # figure 5
    uname = ['Entropy', 'MC-dropout', 'Multi-initial', 'FPsDist', 'LatentDist', 'Random']
    print("Draw Figure 5 saved in {}".format(os.path.join(args.save_dir, 'uncertainty_histplots_'+'_'.join(uname)+'.png')))
    draw_histplot(result_df, uname=uname, args=args)


    # figure 6A
    uname = ['Entropy', 'MC-dropout', 'Ensemble']
    print("Draw Figure 6A saved in {}".format(
        os.path.join(args.save_dir, 'uncertainty_lineplots_' + '_'.join(uname) + '.png')))
    draw_lineplot(result_df, uname=uname, args=args)

    # figure 6B
    uname = ['Ensemble']
    print("Draw Figure 6B saved in {}".format(
        os.path.join(args.save_dir, 'uncertainty_histplot_' + '_'.join(uname) + '.png')))
    draw_histplot(result_df, uname=uname, args=args)



