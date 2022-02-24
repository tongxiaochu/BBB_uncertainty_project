import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from benchmark_ml import scoring


plt.rcParams['axes.unicode_minus'] = False
sns.set_style('darkgrid', {'font.sans-serif':['SimHei', 'Arial']})

import warnings
warnings.filterwarnings('ignore')

metric_ls = ['roc_auc', 'prc_auc', 'matthews_corrcoef', 'balanced_accuracy']

def histplot_for_model_performance():
    model_name = ['BBB score', 'RF(ECFP)', 'MLP(ECFP)', 'RF(PCP)', 'MLP(PCP)', 'AttentiveFP', 'GROVER']
    models_comparsion_metrics_df = pd.DataFrame(index=['ROC_AUC', 'PRC_AUC', 'MCC', 'BACC'])
    for model in model_name:
        result_dir = os.path.join('../BBBp_results/', model, 'test_result.csv')
        test_result = pd.read_csv(result_dir)
        if model == 'BBB score':
            test_score = scoring(test_result['Target'], test_result['Prediction'], ['roc_auc', 'prc_auc'])
            test_score.update(scoring(test_result['Target'], test_result['Prediction_label'], ['matthews_corrcoef', 'balanced_accuracy']))
        else:
            test_score = scoring(test_result['Target'], test_result['Prediction'], metric_ls)
        models_comparsion_metrics_df[model] = test_score.values()
    models_comparsion_metrics_df = round(models_comparsion_metrics_df, 4)
    print(models_comparsion_metrics_df)
    models_comparsion_metrics_data_list = []
    for metric in models_comparsion_metrics_df.index:
        for model in models_comparsion_metrics_df.columns:
            models_comparsion_metrics_data_list.append([metric, model, models_comparsion_metrics_df.loc[metric, model]])
    models_comparision_metrics_data_df = pd.DataFrame(models_comparsion_metrics_data_list, columns=['metric', 'model', 'score'])

    sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)
    fig, ax = plt.subplots(1, 1, figsize=(20, 9))
    sns.barplot(
        axes=ax,
        x='metric', y='score', data=models_comparision_metrics_data_df, hue = 'model',
        palette="RdBu",  alpha=1, saturation=0.75,
                )
    # ax.set_title('Models comparison metrics', size=22)
    ax.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.set_xlabel('metric', size=22)
    ax.set_ylabel('score', size=22)
    ax.set_ylim([0.3, 1])
    ax.tick_params(labelsize=20)
    plt.setp(ax.get_legend().get_texts(), fontsize='22')
    plt.savefig('../BBBp_results/models_comparison_metrics.png', bbox_inches='tight',
                        format='png')



def histplot_for_uncertainty():
    uncertainty_under_curve_area_df = pd.read_csv('../BBBp_results/uncertainty_under_curve_area.csv', index_col=0)
    uncertainty_under_curve_area_data_list = []
    for metric in uncertainty_under_curve_area_df.index:
        for model in uncertainty_under_curve_area_df.columns:
            uncertainty_under_curve_area_data_list.append([metric, model, uncertainty_under_curve_area_df.loc[metric, model]])
    models_comparision_metrics_data_df = pd.DataFrame(uncertainty_under_curve_area_data_list, columns=['uncertainty', 'model', 'score'])

    sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)
    fig, ax = plt.subplots(1, 1, figsize=(20, 9))
    sns.barplot(
        axes=ax,
        x='uncertainty', y='score', data=models_comparision_metrics_data_df, hue='model',
        palette=[sns.color_palette("RdBu_r", 7)[i] for i in [0, 1, -2, -1]],  alpha=1, saturation=0.75,
                )
    ax.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.set_xlabel('uncertainty', size=22)
    ax.set_ylabel('MCC_AUC', size=22)
    ax.set_ylim([0.4, 0.9])
    ax.tick_params(labelsize=20)
    plt.setp(ax.get_legend().get_texts(), fontsize='22')
    plt.savefig('../BBBp_results/uncertainty_under_curve_area.png', bbox_inches='tight',
                        format='png')


if __name__ == '__main__':

    histplot_for_model_performance()

