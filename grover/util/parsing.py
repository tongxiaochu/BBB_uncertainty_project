import json
import os
import pickle
from argparse import ArgumentParser, Namespace
from tempfile import TemporaryDirectory

import torch

from grover.data.molfeaturegenerator import get_available_features_generators
from grover.util.utils import makedirs


def add_common_args(parser: ArgumentParser):
    parser.add_argument('--no_cache', action='store_true', default=True,
                        help='Turn off caching mol2graph computation')
    parser.add_argument('--gpu', type=int,
                        choices=list(range(torch.cuda.device_count())),
                        help='Which GPU to use')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Turn off cuda')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')

def add_predict_args(parser: ArgumentParser):
    """
    Adds predict arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    add_common_args(parser)

    parser.add_argument('--input', type=str,
                        help='Path to CSV file containing testing data for which predictions will be made')
    parser.add_argument('--use_compound_names', action='store_true', default=False,
                        help='Use when test data file contains compound names in addition to SMILES strings')
    parser.add_argument('--output', type=str,
                        help='Path to CSV file where predictions will be saved')
    parser.add_argument('--checkpoint_dir', type=str,
                        help='Directory from which to load model checkpoints'
                             '(walks directory and ensembles all models that are found)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to model checkpoint (.pt file)')

    parser.add_argument('--features_generator', type=str, nargs='*',
                        choices=get_available_features_generators(),
                        help='Method of generating additional features')
    parser.add_argument('--features_path', type=str, nargs='*',
                        help='Path to features to use in FNN (instead of features_generator)')
    parser.add_argument('--no_features_scaling', action='store_true', default=False,
                        help='Turn off scaling of features')
    parser.add_argument('--max_data_size', type=int,
                        help='Maximum number of data points to load')

def add_fingerprint_args(parser):
    add_common_args(parser)
    # parameters for fingerprints generation
    parser.add_argument('--input', type=str, help='Input csv file which contains SMILES')
    parser.add_argument('--output', type=str,
                        help='Path to npz file where predictions will be saved')
    parser.add_argument('--features_path', type=str, nargs='*',
                        help='Path to features to use in FNN (instead of features_generator)')
    parser.add_argument('--fingerprint_source', type=str,
                        choices=['atom', 'bond', 'both'], default='both',
                        help='The source to generate the fingerprints.')
    parser.add_argument('--checkpoint_path', type=str, help='model path')
    parser.add_argument('--checkpoint_dir', type=str,
                        help='Directory from which to load model checkpoints'
                             '(walks directory and ensembles all models that are found)')


def add_train_args(parser: ArgumentParser):
    """
    Adds training arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    add_common_args(parser)
    # General arguments
    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV file')
    parser.add_argument('--use_compound_names', action='store_true', default=False,
                        help='Use when test data file contains compound names in addition to SMILES strings')
    parser.add_argument('--max_data_size', type=int,
                        help='Maximum number of data points to load')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Whether to skip training and only test the model')
    parser.add_argument('--features_only', action='store_true', default=False,
                        help='Use only the additional features in an FFN, no graph network')
    parser.add_argument('--features_generator', type=str, nargs='*',
                        choices=get_available_features_generators(),
                        help='Method of generating additional features')
    parser.add_argument('--features_path', type=str, nargs='*',
                        help='Path to features to use in FNN (instead of features_generator)')
    parser.add_argument('--num_bits', type=int, default=2048,
                        help='Number of bits in morgan fingerprint')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--save_smiles_splits', action='store_true', default=False,
                        help='Save smiles for each train/val/test splits for prediction convenience later')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory from which to load model checkpoints'
                             '(walks directory and ensembles all models that are found)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--dataset_type', type=str,
                        choices=['classification', 'regression'], default='classification',
                        help='Type of dataset, e.g. classification or regression.'
                             'This determines the loss function used during training.')
    parser.add_argument('--separate_val_path', type=str,
                        help='Path to separate val set, optional')
    parser.add_argument('--separate_val_features_path', type=str, nargs='*',
                        help='Path to file with features for separate val set')
    parser.add_argument('--separate_test_path', type=str,
                        help='Path to separate test set, optional')
    parser.add_argument('--separate_test_features_path', type=str, nargs='*',
                        help='Path to file with features for separate test set')
    parser.add_argument('--split_type', type=str, default='random',
                        choices=['random', 'scaffold_balanced', 'predetermined', 'crossval', 'index_predetermined'],
                        help='Method of splitting the data into train/val/test')
    parser.add_argument('--split_sizes', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        help='Split proportions for train/validation/test sets')
    parser.add_argument('--num_folds', type=int, default=1,
                        help='Number of folds when performing cross validation')
    parser.add_argument('--folds_file', type=str, default=None,
                        help='Optional file of fold labels')
    parser.add_argument('--val_fold_index', type=int, default=None,
                        help='Which fold to use as val for leave-one-out cross val')
    parser.add_argument('--test_fold_index', type=int, default=None,
                        help='Which fold to use as test for leave-one-out cross val')
    parser.add_argument('--crossval_index_dir', type=str, 
                        help='Directory in which to find cross validation index files')
    parser.add_argument('--crossval_index_file', type=str, 
                        help='Indices of files to use as train/val/test'
                             'Overrides --num_folds and --seed.')
    parser.add_argument('--seed', type=int, default=921013,
                        help='Random seed to use when splitting data into train/val/test sets.'
                             'When `num_folds` > 1, the first fold uses this seed and all'
                             'subsequent folds add 1 to the seed.')
    parser.add_argument('--metric', type=str, default=None,
                        choices=['roc_auc',
                                 'prc_auc',
                                 'rmse',
                                 'mae',
                                 'r2',
                                 'accuracy',
                                 'recall',
                                 'sensitivity',
                                 'specificity',
                                 'matthews_corrcoef'],
                        help='Metric to use during evaluation.'
                             'Note: Does NOT affect loss function used during training'
                             '(loss is determined by the `dataset_type` argument).'
                             'Note: Defaults to "roc_auc" for classification and "rmse" for regression.')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Skip non-essential print statements')
    parser.add_argument('--log_frequency', type=int, default=10,
                        help='The number of batches between each logging of the training loss')

    parser.add_argument('--show_individual_scores', action='store_true', default=False,
                        help='Show all scores for individual targets, not just average, at the end')

    parser.add_argument('--config_path', type=str,
                        help='Path to a .json file containing arguments. Any arguments present in the config'
                             'file will override arguments specified via the command line or by the defaults.')
    parser.add_argument('--debug', action='store_true', default=False, help='Set number of workers to 0 to help debug')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Add tensorboard logger')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to task')
    parser.add_argument('--warmup_epochs', type=float, default=2.0,
                        help='Number of epochs during which learning rate increases linearly from'
                             'init_lr to max_lr. Afterwards, learning rate decreases exponentially'
                             'from max_lr to final_lr.')
    parser.add_argument('--init_lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-3,
                        help='Maximum learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4,
                        help='Final learning rate')
    parser.add_argument('--no_features_scaling', action='store_true', default=False,
                        help='Turn off scaling of features')
    parser.add_argument('--early_stop_epoch', type=int, default=1000, help='If val loss did not drop in '
                                                                         'this epochs, stop running')

    # Model arguments
    parser.add_argument('--model_type', default="dualmpnn", help="Choose the model: 'mpnn', 'multimpnn', 'dualmpnn' are"
                                                                 " available.")
    parser.add_argument('--ensemble_size', type=int, default=1,
                        help='Number of models in ensemble')
    parser.add_argument('--input_layer', type=str, default='fc',
                        choices=['fc', 'gcn'],
                        help='The input layer of model')
    parser.add_argument('--hidden_size', type=int, default=3,
                        help='Dimensionality of hidden layers in MPN')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='Whether to add bias to linear layers')
    parser.add_argument('--depth', type=int, default=3,
                        help='Number of message passing steps')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout probability')
    parser.add_argument('--activation', type=str, default='ReLU',
                        choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'],
                        help='Activation function')
    parser.add_argument('--undirected', action='store_true', default=False,
                        help='Undirected edges (always sum the two relevant bond vectors)')                     
    parser.add_argument('--ffn_hidden_size', type=int, default=None,
                        help='Hidden dim for higher-capacity FFN (defaults to hidden_size)')
    parser.add_argument('--ffn_num_layers', type=int, default=2,
                        help='Number of layers in FFN after MPN encoding')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight_decay')
    parser.add_argument('--select_by_loss', action='store_true', default=False,
                        help='Use validation loss as refence standard to select best model to predict')
    parser.add_argument('--skip_epoch', type=int, default=0, help='Start select the model after skip_epoch '
                                                                  'in training. Default is 0.')

    parser.add_argument("--embedding_output_type", default="atom", choices=["atom", "bond", "both"],
                        help="This the model parameters for pretrain model.")

    # self-attention readout.
    parser.add_argument('--self_attention', action='store_true', default=False, help='Use self attention layer')
    parser.add_argument('--attn_hidden', type=int, default=4, nargs='?', help='self attention layer hidden layer size')
    parser.add_argument('--attn_out', type=int, default=128, nargs='?', help='self attention layer output feature size')

    parser.add_argument('--dense', action='store_true', default=False, help='Cat message in different time stamp in mpn')
    parser.add_argument('--bond_drop_rate', type=float, default=0, help='Drop out bond in molecular')
    parser.add_argument('--distinct_init', action='store_true', default=False,
                        help='Using distinct weight init for model ensemble')


    parser.add_argument('--aug_rate', type=float, default=0, help='On the fly smiles enumeration')
    #parser.add_argument('--fine_tune_fc', action='store_true', default=False,
    #                    help='Enable fine tune model. (Fix other layer, only fine tune fc layer.)')
    parser.add_argument('--fine_tune_coff', type=float, default=1,
                        help='Enable distinct fine tune for fc and other layer')

    # Arguments for DualMPNN
    parser.add_argument('--dist_coff', type=float, default=0.1, help='The dist coefficient for the DualMPNN.')
    parser.add_argument('--no_attach_fea', action='store_true', default=False,
                        help="Do not attach feature in message passing.")

    parser.add_argument('--output_feature', action='store_true', default=False,
                        help='Get features as output')

    # parameter for GROVER.
    parser.add_argument('--num_attn_head', type=int, default=1, help='The attention head in MTBlock.')
    parser.add_argument('--num_mt_block', type=int, default=1, help="The number of MTBlock.")

    # For multi-gpu finetune.
    parser.add_argument('--enbl_multi_gpu', dest='enbl_multi_gpu',
                        action='store_true', default=False,
                        help='enable multi-GPU training')

    # For uncertainty estimation.
    parser.add_argument('--pred_times', type=int, default=20, help="The number of predictions for MCdropout.")

def update_checkpoint_args(args: Namespace):
    """
    Walks the checkpoint directory to find all checkpoints, updating args.checkpoint_paths and args.ensemble_size.

    :param args: Arguments.
    """
    if hasattr(args, 'checkpoint_paths') and args.checkpoint_paths is not None:
        return

    if args.checkpoint_dir is not None and args.checkpoint_path is not None:
        raise ValueError('Only one of checkpoint_dir and checkpoint_path can be specified.')

    if args.checkpoint_dir is None:
        args.checkpoint_paths = [args.checkpoint_path] if args.checkpoint_path is not None else None
        return

    args.checkpoint_paths = []

    for root, _, files in os.walk(args.checkpoint_dir):
        for fname in files:
            if fname.endswith('.pt'):
                args.checkpoint_paths.append(os.path.join(root, fname))

    args.ensemble_size = len(args.checkpoint_paths)

    if args.ensemble_size == 0:
        raise ValueError(f'Failed to find any model checkpoints in directory "{args.checkpoint_dir}"')


def modify_predict_args(args: Namespace):
    """
    Modifies and validates predicting args in place.

    :param args: Arguments.
    """
    assert args.input
    assert args.output
    assert args.checkpoint_dir is not None or args.checkpoint_path is not None or args.checkpoint_paths is not None

    update_checkpoint_args(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    del args.no_cuda

    # Create directory for preds path
    makedirs(args.output, isfile=True)
    setattr(args, 'fingerprint', False)


def modify_fingerprint_args(args):
    assert args.input
    assert args.output
    assert args.checkpoint_dir is not None or args.checkpoint_path is not None or args.checkpoint_paths is not None

    update_checkpoint_args(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    del args.no_cuda
    makedirs(args.output, isfile=True)
    setattr(args, 'fingerprint', True)


def get_newest_train_args():
    """
    For backward compatibility.

    :return:  A Namespace containing the newest training arguments
    """
    dummy_parser = ArgumentParser()
    add_train_args(dummy_parser)
    args = dummy_parser.parse_args(args=[])
    args.data_path = ''
    modify_train_args(args)
    return args

def modify_train_args(args: Namespace):
    """
    Modifies and validates training arguments in place.

    :param args: Arguments.
    """
    global temp_dir  # Prevents the temporary directory from being deleted upon function return

    # Load config file
    if args.config_path is not None:
        with open(args.config_path) as f:
            config = json.load(f)
            for key, value in config.items():
                if 'features_path' in key:
                    value = value.split(',')
                setattr(args, key, value)

    assert args.data_path is not None
    assert args.dataset_type is not None

    if args.save_dir is not None:
        makedirs(args.save_dir)
    else:
        temp_dir = TemporaryDirectory()
        args.save_dir = temp_dir.name

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    del args.no_cuda

    args.features_scaling = not args.no_features_scaling
    del args.no_features_scaling

    if args.metric is None:
        if args.dataset_type == 'classification':
            args.metric = 'roc_auc'
        else:
            args.metric = 'rmse'

    if not ((args.dataset_type == 'classification' and args.metric in ['roc_auc', 'prc_auc', 'accuracy']) or
            (args.dataset_type == 'regression' and args.metric in ['rmse', 'mae', 'r2']) or
            (args.dataset_type == 'multiclass' and args.metric in ['accuracy', 'roc_auc', 'prc_auc'])):
        raise ValueError(f'Metric "{args.metric}" invalid for dataset type "{args.dataset_type}".')

    args.minimize_score = args.metric in ['rmse', 'mae']

    update_checkpoint_args(args)
    
    if args.features_only:
        assert args.features_generator or args.features_path

    args.use_input_features = args.features_generator or args.features_path

    if args.features_generator is not None and 'rdkit_2d_normalized' in args.features_generator:
        assert not args.features_scaling


    args.num_lrs = 1

    if args.ffn_hidden_size is None:
        args.ffn_hidden_size = args.hidden_size

    assert (args.split_type == 'predetermined') == (args.folds_file is not None) == (args.test_fold_index is not None)
    assert (args.split_type == 'crossval') == (args.crossval_index_dir is not None)
    assert (args.split_type in ['crossval', 'index_predetermined']) == (args.crossval_index_file is not None)
    if args.split_type in ['crossval', 'index_predetermined']:
        with open(args.crossval_index_file, 'rb') as rf:
            args.crossval_index_sets = pickle.load(rf)
        args.num_folds = len(args.crossval_index_sets)
        args.seed = 0

    if args.test:
        args.epochs = 0

    if args.bond_drop_rate > 0:
        args.no_cache = True

    setattr(args, 'fingerprint', False)


def parse_args() -> Namespace:
    """
    Parses arguments for training and testing (includes modifying/validating arguments).

    :return: A Namespace containing the parsed, modified, and validated args.
    """
    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()
    modify_train_args(args)

    return args

def add_GROVER_args(parser: ArgumentParser):

    add_train_args(parser)


def add_MLP_args(parser: ArgumentParser):
    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV file')
    parser.add_argument('--separate_test_path', type=str,
                        help='Path to separate test set, optional')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Skip non-essential print statements')
    parser.add_argument('--seed', type=int, default=921013,
                        help='Random seed')
    parser.add_argument('--ensemble_size', type=int, default=1,
                        help='Number of models in ensemble')
    parser.add_argument('--feature_type', type=str, default='PCP',
                        help='features for ML models [PCP, ECFP]')
    parser.add_argument('--dataset_type', type=str,
                        choices=['classification', 'regression'], default='classification',
                        help='Type of dataset, e.g. classification or regression.'
                             'This determines the loss function used during training.')


def add_RF_args(parser: ArgumentParser):
    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV file')
    parser.add_argument('--separate_test_path', type=str,
                        help='Path to separate test set, optional')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Skip non-essential print statements')
    parser.add_argument('--seed', type=int, default=921013,
                        help='Random seed')
    parser.add_argument('--ensemble_size', type=int, default=1,
                        help='Number of models in ensemble')
    parser.add_argument('--feature_type', type=str, default='PCP',
                        help='features for ML models [PCP, ECFP]')
    parser.add_argument('--dataset_type', type=str,
                        choices=['classification', 'regression'], default='classification',
                        help='Type of dataset, e.g. classification or regression.'
                             'This determines the loss function used during training.')

def add_AttentiveFP_args(parser: ArgumentParser):
    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV file')
    parser.add_argument('--separate_test_path', type=str,
                        help='Path to separate test set, optional')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Skip non-essential print statements')
    parser.add_argument('--seed', type=int, default=921013,
                        help='Random seed')
    parser.add_argument('--ensemble_size', type=int, default=1,
                        help='Number of models in ensemble')
    parser.add_argument('--dataset_type', type=str,
                        choices=['classification', 'regression'], default='classification',
                        help='Type of dataset, e.g. classification or regression.'
                             'This determines the loss function used during training.')

    # For uncertainty estimation.
    parser.add_argument('--pred_times', type=int, default=20, help="The number of predictions for MCdropout.")

