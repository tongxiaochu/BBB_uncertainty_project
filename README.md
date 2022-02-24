# BBB_uncertainty_project
code for "Blood–Brain Barrier Penetration Prediction Enhanced by Uncertainty Estimation"

#### process test dataset and save features
```
python save_features.py --data_path ../dataset/Sdata-process-flow-step5-testdata.csv
                        --features_generator rdkit_2d_normalized
                        --save_path ../dataset/Sdata-process-flow-step5-testdata.npz
                        --sequential
```

#### estimate uncertainty for S-data by GROVER
```
python main.py GROVER --seed 921013
                      --data_path dataset/MoleculeNet-BBBP-process-flow-step5-traindata.csv
                      --features_path dataset/MoleculeNet-BBBP-process-flow-step5-traindata.npz
                      --separate_test_path dataset/Sdata-process-flow-step5-testdata.csv
                      --separate_test_features_path dataset/Sdata-process-flow-step5-testdata.npz
                      --dataset_type classification
                      --no_features_scaling
                      --ensemble_size 5
                      --num_folds 1
                      --epochs 60
                      --dropout 0.5
                      --batch_size 64
                      --activation PReLU
                      --attn_hidden 16
                      --attn_out 4
                      --aug_rate 0
                      --depth 6
                      --dist_coff 0.3
                      --ffn_hidden_size 10
                      --ffn_num_layers 3
                      --final_lr 8e-05
                      --init_lr 0.00008
                      --max_lr 0.0008
                      --hidden_size 8
                      --num_attn_head 4
                      --weight_decay 1e-7
                      --pred_times 100
                      --gpu 0
                      --save_dir ./BBBp_results/GROVER
```

#### estimate uncertainty for S-data by AttentiveFP
```
python main.py AttentiveFP --seed 921013
                   --data_path dataset/MoleculeNet-BBBP-process-flow-step5-traindata.csv
                   --separate_test_path dataset/Sdata-process-flow-step5-testdata.csv
                   --dataset_type classification
                   --ensemble_size 5
                   --pred_times 100
                   --save_dir ./BBBp_results/AttentiveFP
```

#### estimate uncertainty for S-data by RL/MLP
```
python main.py MLP --seed 921013
                   --data_path dataset/MoleculeNet-BBBP-process-flow-step5-traindata.csv
                   --separate_test_path dataset/Sdata-process-flow-step5-testdata.csv
                   --feature_type PCP
                   --dataset_type classification
                   --ensemble_size 5
                   --save_dir ./BBBp_results/MLP(PCP)
```
