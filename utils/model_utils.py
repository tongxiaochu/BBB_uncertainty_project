from typing import Tuple, Dict, List, Union, Any
from argparse import Namespace
from logging import Logger

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from grover.model.models import GroverFinetuneTask

from grover.util.nn_utils import initialize_weights, NoamLR
from grover.util.utils import build_optimizer, build_lr_scheduler, makedirs, get_loss_func, \
    save_checkpoint, load_checkpoint, build_model
from utils.dataset_utils import MyDataset, GraphCollator

Model = Union[GroverFinetuneTask]


class MidFeature(torch.nn.Module):
    def __init__(self, network):
        super(MidFeature, self).__init__()
        self.network = network
        net_list = list(network.children())
        self.atom_features_ffn = net_list[-2][0][:8]
        self.bond_features_ffn = net_list[-2][1][:8]

    def forward(self, batch, features_batch):
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch
        mpnn = self.network.grover(batch)
        # Share readout
        mol_atom_from_bond_output = self.network.readout(mpnn["atom_from_bond"], a_scope)
        mol_atom_from_atom_output = self.network.readout(mpnn["atom_from_atom"], a_scope)

        features_batch = torch.from_numpy(np.stack(features_batch)).float()
        features_batch = features_batch.to(mpnn["atom_from_atom"])
        if len(features_batch.shape) == 1:
            features_batch = features_batch.view([1, features_batch.shape[0]])

        mol_atom_from_atom_output = torch.cat([mol_atom_from_atom_output, features_batch], 1)
        mol_atom_from_bond_output = torch.cat([mol_atom_from_bond_output, features_batch], 1)

        atom_ffn_output = self.atom_features_ffn(mol_atom_from_atom_output)
        bond_ffn_output = self.bond_features_ffn(mol_atom_from_bond_output)
        return torch.cat((atom_ffn_output, bond_ffn_output), 1)


class FitModule:
    def __init__(self, model_idx, scaler, args: Namespace, logger: Logger = None):
        self.model_idx = model_idx
        self.scaler = scaler
        self.args = args
        self.logger = logger

    def init_model(self):
        if self.logger is not None:
            debug, info = self.logger.debug, self.logger.info
        else:
            debug = info = print

        # Load/build model
        if self.args.checkpoint_paths is not None:
            if len(self.args.checkpoint_paths) == 1:
                cur_model = 0
            else:
                cur_model = self.model_idx
            debug(f'Loading model {cur_model} from {self.args.checkpoint_paths[cur_model]}')
            self.model = load_checkpoint(self.args.checkpoint_paths[cur_model], current_args=self.args, logger=self.logger)
        else:
            debug(f'Building model {self.model_idx}')
            self.model = build_model(model_idx=self.model_idx, args=self.args)

        if self.args.fine_tune_coff != 1 and self.args.checkpoint_paths is not None:
            debug("Fine tune fc layer with different lr")
            initialize_weights(model_idx=self.model_idx, model=self.model.ffn, distinct_init=self.args.distinct_init)

        # Get loss and metric functions
        self.loss_func = get_loss_func(self.args, self.model)
        self.optimizer = build_optimizer(self.model, self.args)

        # debug(self.model)
        # debug(f'Number of parameters = {param_count(self.model):,}')
        if self.args.cuda:
            debug('Moving model to cuda')
            self.model = self.model.cuda()

        # Learning rate schedulers
        self.scheduler = build_lr_scheduler(self.optimizer, self.args)

    def get_loader(
            self,
            X: np.array,
            y: np.array = None,
            masks: np.array = None,
            batch_size: int = 1,
            shuffle: bool = False
    ):
        """Convert X and y Tensors to a DataLoader
            If y is None, use a dummy Tensor
        """
        if y is None:
            y = torch.zeros(X.shape[0])
        if masks is None:
            masks = torch.zeros(X.shape[0])
        data = MyDataset(X, y, masks)
        loader = DataLoader(data, batch_size, shuffle, collate_fn=GraphCollator(self.args))
        return loader

    def fit(self,
            X: np.array,
            y: np.array,
            **fit_kwargs
            ):

        masks = fit_kwargs['masks']
        debug = self.logger.debug if self.logger is not None else print
        args = self.args
        self.init_model()

        optimizer = self.optimizer
        loss_func = self.loss_func
        scheduler = self.scheduler

        cum_loss_sum, cum_iter_count = 0, 0
        data_loader = self.get_loader(X, y, masks, args.batch_size, shuffle=True)
        # Run training loop
        for t in range(args.epochs):
            for i, item in enumerate(data_loader):
                batch, features_batch, targets, mask = item
                if self.args.cuda:
                    mask, targets = mask.cuda(), targets.cuda()
                class_weights = torch.ones(targets.shape)
                if self.args.cuda:
                    class_weights = class_weights.cuda()
                # Backprop
                optimizer.zero_grad()
                y_batch_pred = self.model(batch, features_batch)
                batch_loss = loss_func(y_batch_pred, targets) * class_weights * mask
                batch_loss = batch_loss.sum() / mask.sum()

                cum_loss_sum += batch_loss.item()
                cum_iter_count += 1

                batch_loss.backward()
                optimizer.step()

                if isinstance(scheduler, NoamLR):
                    scheduler.step()
            if t % 5 == 0 or t == args.epochs-1:
                debug(f'{t}/{args.epochs} epochs， loss {cum_loss_sum / cum_iter_count:.4}')
        return cum_loss_sum / cum_iter_count

    def predict_proba(self, X: np.array) -> np.array:
        """Generates output predictions for the input samples.
        Computation is done in batches.
        # Arguments
            X: input data Tensor.
            batch_size: integer.
        # Returns
            prediction Tensor.
        """
        args = self.args
        if self.args.cuda:
            self.model = self.model.cuda()

        # Build DataLoader
        data_loader = self.get_loader(X, None, None, batch_size=args.batch_size)
        # Batch prediction
        self.model.eval()
        preds = []

        for i, item in enumerate(data_loader):
            batch, features_batch, targets, mask = item

            with torch.no_grad():
                batch_preds = self.model(batch, features_batch)
                if args.fingerprint:
                    preds.extend(batch_preds.data.cpu().numpy())
                    continue
            # Collect vectors
            batch_preds = batch_preds.data.cpu().numpy().tolist()
            if self.scaler is not None:
                batch_preds = self.scaler.inverse_transform(batch_preds)
            preds.extend([[1-p[0], p[0]]for p in batch_preds])

        return np.array(preds)

    def predict(self, X: np.array) -> np.array:
        preds = self.predict_proba(X)
        hard_preds = preds[:, 1].reshape(-1, 1)

        return hard_preds

    def state_dict(self):
        return self.model.state_dict()

    def predict_Ttimes(self, X, T=30):
        args = self.args
        debug = self.logger.debug if self.logger is not None else print

        # Build DataLoader
        data_loader = self.get_loader(X, None, None, batch_size=args.batch_size)
        # Batch prediction
        self.model.eval()
        self.model.apply(self.apply_dropout)
        # predict stochastic dropout model T times
        preds_times = []
        for t in range(T):
            if t % 10 == 0: debug(f'Have predicted for {t+1}/{T} times')
            preds = []
            for i, item in enumerate(data_loader):
                batch, features_batch, targets, mask = item

                with torch.no_grad():
                    batch_preds = self.model(batch, features_batch)
                    if args.fingerprint:
                        preds.extend(batch_preds.data.cpu().numpy())
                        continue
                # Collect vectors
                batch_preds = batch_preds.data.cpu().numpy().tolist()
                if self.scaler is not None:
                    batch_preds = self.scaler.inverse_transform(batch_preds)
                preds.extend(batch_preds)

            preds_times.append([p[0] for p in preds])
        p_hat = np.array(preds_times)
        p_hat_binary = np.array([[[1 - p, p] for p in sample] for sample in p_hat])
        return p_hat_binary

    @staticmethod
    def apply_dropout(m):
        if type(m) == torch.nn.Dropout:
            m.train()
