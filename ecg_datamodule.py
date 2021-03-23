import os
from typing import Optional, Sequence
from warnings import warn

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from clinical_ts.simclr_dataset_wrapper import SimCLRDataSetWrapper


class ECGDataModule(LightningDataModule):

    name = 'ecg_dataset'
    extra_args = {}

    def __init__(
            self,
            config,
            transformations_str,
            t_params, 
            data_dir: str = None,
            val_split: int = 5000,
            num_workers: int = 16,
            batch_size: int = 32,
            seed: int = 42,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.dims = (12, 250)
        # self.val_split = val_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        # self.num_samples = 60000 - val_split

        # self.DATASET = SimCLRDataSetWrapper(
        #    config['eval_batch_size'], **config['eval_dataset'])
        # self.train_loader, self.valid_loader = self.DATASET.get_data_loaders()
        self.config = config
        self.transformations_str = transformations_str
        self.t_params = t_params
        self.set_params()

    def set_params(self):
        dataset = SimCLRDataSetWrapper(
            self.config['batch_size'], **self.config['dataset'], transformations=self.transformations_str, t_params=self.t_params)
        train_loader, valid_loader = dataset.get_data_loaders() 
        self.num_samples = dataset.train_ds_size
        self.transformations = dataset.transformations
    @property
    def num_classes(self):
        """
        Return:
            10
        """
        return 5

    def prepare_data(self):
        pass

    def train_dataloader(self):
        dataset = SimCLRDataSetWrapper(
            self.config['batch_size'], **self.config['dataset'], transformations=self.transformations_str, t_params=self.t_params)
        train_loader, _ = dataset.get_data_loaders()
        return train_loader

    def val_dataloader(self):
        dataset = SimCLRDataSetWrapper(
            self.config['eval_batch_size'], **self.config['eval_dataset'], transformations=self.transformations_str, t_params=self.t_params)
        _, valid_loader_self = dataset.get_data_loaders()
        dataset = SimCLRDataSetWrapper(
            self.config['eval_batch_size'], **self.config['eval_dataset'], transformations=self.transformations_str, t_params=self.t_params, mode="linear_evaluation")
        valid_loader_sup, test_loader_sup = dataset.get_data_loaders()
        # return valid_loader
        return [valid_loader_self, valid_loader_sup, test_loader_sup]


    def test_dataloader(self):
        return self.valid_loader

    def default_transforms(self):
        pass
