#!/bin/bash

TRAIN_DATA="MoleculeNet-BBBP-process-flow-step5-traindata"
TEST_DATA="Sdata-process-flow-step5-testdata"

FEATURES_GENERATOR="rdkit_2d_normalized"
RESULT_DIR=results/BBBP_cls/BBB_finetune

# process dataset
python task/save_features.py --data_path dataset/${TRAIN_DATA}.csv \
                        --features_generator ${FEATURES_GENERATOR} \
                        --save_path dataset/${TRAIN_DATA}.npz --sequential &&

python task/save_features.py --data_path dataset/${TEST_DATA}.csv \
                        --features_generator ${FEATURES_GENERATOR} \
                        --save_path dataset/${TEST_DATA}.npz --sequential &&

# esimate uncertainty
python main.py   --seed 921013 \
                           --data_path dataset/${TRAIN_DATA}.csv \
                           --features_path dataset/${TRAIN_DATA}.npz \
                           --separate_test_path dataset/${TEST_DATA}.csv \
                           --separate_test_features_path dataset/${TEST_DATA}.npz \
                           --dataset_type classification \
                           --no_features_scaling \
                           --ensemble_size 5 --num_folds 1 --epochs 60 \
                           --dropout 0.5  --batch_size  64 --activation PReLU \
                           --attn_hidden 16 --attn_out 4 --aug_rate 0 --depth 6 --dist_coff 0.3 \
                           --ffn_hidden_size 10 --ffn_num_layers 3 --final_lr 8e-05 --init_lr 0.00008 --max_lr 0.0008 \
                           --hidden_size 8 --num_attn_head 4 --weight_decay 1e-7 \
                           --pred_times 100 \
                           --gpu 0 --save_dir ./${RESULT_DIR}  &&

python task/uncertainty_analysis_plot.py --save_dir ./${RESULT_DIR}
