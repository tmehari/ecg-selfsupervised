from .create_logger import create_logger
import numpy as np
from torch.utils.data import DataLoader
# from .customDataLoader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset
from torchvision import datasets
from functools import partial
from pathlib import Path
import pandas as pd
import pdb
try:
    import pickle5 as pickle
except ImportError as e:
    import pickle
from .timeseries_utils import TimeseriesDatasetCrops, reformat_as_memmap, load_dataset
from .ecg_utils import *
from .timeseries_transformations import GaussianNoise, RandomResizedCrop, ChannelResize, Negation, DynamicTimeWarp, DownSample, TimeWarp, TimeOut, ToTensor, BaselineWander, PowerlineNoise, EMNoise, BaselineShift, TGaussianNoise, TRandomResizedCrop, TChannelResize, TNegation, TDynamicTimeWarp, TDownSample, TTimeOut, TBaselineWander, TPowerlineNoise, TEMNoise, TBaselineShift, TGaussianBlur1d, TNormalize, Transpose


logger = create_logger(__name__)


def transformations_from_strings(transformations, t_params):
    if transformations is None:
        return [ToTensor()]

    def str_to_trafo(trafo):
        if trafo == "RandomResizedCrop":
            return TRandomResizedCrop(crop_ratio_range=t_params["rr_crop_ratio_range"], output_size=t_params["output_size"])
        elif trafo == "ChannelResize":
            return TChannelResize(magnitude_range=t_params["magnitude_range"])
        elif trafo == "Negation":
            return TNegation()
        elif trafo == "DynamicTimeWarp":
            return TDynamicTimeWarp(warps=t_params["warps"], radius=t_params["radius"])
        elif trafo == "DownSample":
            return TDownSample(downsample_ratio=t_params["downsample_ratio"])
        elif trafo == "TimeWarp":
            return TimeWarp(epsilon=t_params["epsilon"])
        elif trafo == "TimeOut":
            return TTimeOut(crop_ratio_range=t_params["to_crop_ratio_range"])
        elif trafo == "GaussianNoise":
            return TGaussianNoise(scale=t_params["gaussian_scale"])
        elif trafo == "BaselineWander":
            return TBaselineWander(Cmax=t_params["bw_cmax"])
        elif trafo == "PowerlineNoise":
            return TPowerlineNoise(Cmax=t_params["pl_cmax"])
        elif trafo == "EMNoise":
            return TEMNoise(Cmax=t_params["em_cmax"])
        elif trafo == "BaselineShift":
            return TBaselineShift(Cmax=t_params["bs_cmax"])
        elif trafo == "GaussianBlur":
            return TGaussianBlur1d()
        elif trafo == "Normalize":
            return TNormalize()
        else:
            raise Exception(str(trafo) + " is not a valid transformation")

    # for numpy transformations
    # trafo_list = [str_to_trafo(trafo)
    #               for trafo in transformations] + [ToTensor()]

    # for torch transformations
    trafo_list = [ToTensor(transpose_data=False)] + [str_to_trafo(trafo)
                                                     for trafo in transformations] + [Transpose()]
    return trafo_list


class SimCLRDataSetWrapper(object):

    def __init__(self, batch_size, num_workers, valid_size, input_shape, s, data_folder, target_folders, target_fs, recreate_data_ptb_xl,
                 mode="pretraining", transformations=None, t_params=None, ptb_xl_label="label_diag_superclass", filter_cinc=False, 
                 percentage=1.0, swav=False, nmb_crops=7, folds=8, test=False):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.s = s
        self.input_shape = eval(input_shape)
        self.data_folder = Path(data_folder)
        # Path(target_folder+str(target_fs))
        self.target_folders = [Path(target_folder)
                               for target_folder in target_folders]
        self.target_fs = target_fs
        self.recreate_data_ptb_xl = recreate_data_ptb_xl
        self.val_ds_idmap = None
        self.lbl_itos = None
        self.transformations = transformations_from_strings(
            transformations, t_params)
        self.train_ds_size = 0
        self.val_ds_size = 0
        self.ptb_xl_label = ptb_xl_label
        self.filter_cinc = filter_cinc
        self.percentage = percentage
        self.swav = swav
        self.nmb_crops = nmb_crops
        self.folds = folds
        self.test = test
        if mode in ["linear_evaluation", "pretraining"]:
            self.mode = mode
        else:
            raise("mode unkown")

    def get_data_loaders(self):
        data_augment = self._get_simclr_pipeline_transform()

        # train_dataset = datasets.STL10('./data', split='train+unlabeled', download=True,
        #                                transform=SimCLRDataTransform(data_augment))

        if self.mode == "linear_evaluation":
            # transformations = transforms.Compose([RandomResizedCrop(crop_ratio_range=[0.5, 1.0]),
            #                                  ToTensor()])
            # transformations = data_augment
            # transformations = ToTensor()
            train_ds, val_ds = self._get_datasets(
                self.target_folders[0], transforms=data_augment)
            self.val_ds_idmap = val_ds.get_id_mapping()
        else:
            
            wrapper_transform = SwAVDataTransform(data_augment, num_crops=self.nmb_crops) if self.swav else SimCLRDataTransform(data_augment)
            datasets = [self._get_datasets(target_folder, transforms=wrapper_transform) for target_folder in self.target_folders]
            train_datasets, valid_datasets = list(zip(*datasets))

            train_ds = ConcatDataset(list(train_datasets))
            val_ds = ConcatDataset(list(valid_datasets))

        train_loader, valid_loader = self.get_train_validation_data_loaders(
            train_ds, val_ds)

        self.train_ds_size = len(train_ds)
        self.val_ds_size = len(val_ds)
        return train_loader, valid_loader

    def _get_datasets(self, target_folder, transforms=None):
        logger.info("get dataset from " + str(target_folder))
        # Dataset parameters
        input_channels = 12
        target_fs = 100
        # Training setting
        input_size = 250  # originally 600
        chunkify_train = False
        chunkify_valid = self.mode != "pretraining"
        chunk_length_train = input_size  # target_fs*6
        chunk_length_valid = input_size
        min_chunk_length = input_size  # chunk_length
        stride_length_train = chunk_length_train//4  # chunk_length_train//8
        stride_length_valid = input_size//2  # chunk_length_valid

        copies_valid = 0  # >0 should only be used with chunkify_valid=False
        if self.test:
            valid_fold=10
            test_fold=9
        else:
            valid_fold=9
            test_fold=10
       
        train_folds = []
        train_folds = list(range(1, 11))
        train_folds.remove(test_fold)
        train_folds.remove(valid_fold)
        train_folds = np.array(train_folds)

        df_memmap_filename = "df_memmap.pkl"
        memmap_filename = "memmap.npy"

        # df, lbl_itos,  mean, std = prepare_data_ptb_xl(self.data_folder, min_cnt=50, target_fs=self.target_fs,
        #                                                                        channels=input_channels, channel_stoi=channel_stoi_default, target_folder=self.target_folder, recreate_data=self.recreate_data_ptb_xl)
        df_mapped, lbl_itos, mean, std = load_dataset(target_folder)
        
       
        if(self.recreate_data_ptb_xl):
            df_mapped = reformat_as_memmap(
                df, target_folder/(memmap_filename), data_folder=target_folder)
        else:
            # df_mapped = pd.read_pickle(
            #     target_folder/(df_memmap_filename))
            df_mapped = pickle.load(open(target_folder/(df_memmap_filename), "rb"))
        #self.lbl_itos = np.array(lbl_itos[label])
        
        self.lbl_itos = lbl_itos
        self.num_classes = len(lbl_itos)
        # print("num classes:", self.num_classes)
        
        if "ptb" in str(target_folder):
            label = self.ptb_xl_label  # just possible for ptb xl
            self.lbl_itos = np.array(lbl_itos[label])
            label = label + "_filtered_numeric"
        else:
            label = "label"
            self.lbl_itos = lbl_itos

        df_mapped["diag_label"] = df_mapped[label].copy()
        if "ptb" in str(target_folder) or self.mode == "linear_evaluation":
            logger.debug("get labels for linear evaluation on ptb")
            df_mapped["label"] = df_mapped[label].apply(
                lambda x: multihot_encode(x, len(self.lbl_itos)))
        else:
            logger.debug("insert artifical labels to non-ptb dataset")
            df_mapped["label"] = df_mapped[label].apply(
                lambda x: np.array([1, 0, 0, 0, 0]))

        # logger.info("labels: " + str(self.lbl_itos))
        # df_mapped["label"] = df_mapped["label"].apply(lambda x: onehot_encode(x, len(self.lbl_itos)))

       
        if self.mode == "pretraining":
            valid_fold = test_fold = 9
            if self.percentage < 1.0:
                logger.info("reduce dataset to {}%".format(self.percentage*100))
                total_samples = len(df_mapped)
                num_samples = int(self.percentage*total_samples)
                sample_indices = np.sort(np.random.choice(np.arange(total_samples), size=num_samples, replace=False))
                df_mapped = df_mapped.loc[sample_indices]
        
            df_train = df_mapped[(df_mapped.strat_fold != test_fold) & (
                df_mapped.strat_fold != valid_fold) & (df_mapped.label.apply(lambda x: np.sum(x) > 0))]
        else:
            assert(self.folds < 9)
            df_train = df_mapped[(df_mapped.strat_fold.apply(lambda x: x in train_folds[range(self.folds)]) & (df_mapped.label.apply(lambda x: np.sum(x) > 0)))]
        
        df_valid = df_mapped[(df_mapped.strat_fold == valid_fold) & (
            df_mapped.label.apply(lambda x: np.sum(x) > 0))]
        df_test = df_mapped[(df_mapped.strat_fold == test_fold) & (
            df_mapped.label.apply(lambda x: np.sum(x) > 0))]

        if self.filter_cinc and "cinc" in str(target_folder):
            df_train = filter_out_datasets(df_train)
            df_valid = filter_out_datasets(df_valid)
            df_test = filter_out_datasets(df_test)

        train_ds = TimeseriesDatasetCrops(df_train, input_size, num_classes=len(self.lbl_itos), data_folder=target_folder, chunk_length=chunk_length_train if chunkify_train else 0,
                                          min_chunk_length=min_chunk_length, stride=stride_length_train, transforms=transforms, annotation=False, col_lbl="label", memmap_filename=target_folder/(memmap_filename))
        val_ds = TimeseriesDatasetCrops(df_valid, input_size, num_classes=len(self.lbl_itos), data_folder=target_folder, chunk_length=chunk_length_valid if chunkify_valid else 0,
                                        min_chunk_length=min_chunk_length, stride=stride_length_valid, transforms=transforms, annotation=False, col_lbl="label", memmap_filename=target_folder/(memmap_filename))
        self.df_train = df_train
        self.df_valid = df_valid 
        self.df_test = df_test
        return train_ds, val_ds

    def _get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        # find transformations in ecg_transformations.py file
        # data_transforms = transforms.Compose([RandomResizedCrop(crop_ratio_range=[0.5, 1.0]),
        #                                      ChannelResize(magnitude_range=[0.33, 3]),
        #                                      DynamicTimeWarp(),
        #                                      ToTensor()])
        # data_transforms = [RandomResizedCrop(), ChannelResize(), ToTensor()]
        data_transforms = transforms.Compose(self.transformations)
        return data_transforms

    def get_train_validation_data_loaders(self, train_ds, val_ds):

        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  num_workers=self.num_workers, pin_memory=True, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size,
                                shuffle=False, num_workers=self.num_workers, pin_memory=True)

        return train_loader, val_loader

class SimCLRDataTransform(object):
    def __init__(self, transform):
        if transform is None:
            self.transform = lambda x: x
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj

class SwAVDataTransform(object):
    def __init__(self, transform, num_crops=7):
        if transform is None:
            self.transform = lambda x: x
        self.transform = transform
        self.num_crops=num_crops

    def __call__(self, sample):
        transformed = [] 
        for _ in range(self.num_crops):
            transformed.append(self.transform(sample)[0])
        return transformed, sample[1]


def multihot_encode(x, num_classes):
    res = np.zeros(num_classes, dtype=np.float32)
    res[x] = 1
    return res


def filter_out_datasets(df, negative_datasets={"PTB", "PTB-XL"}):
    datasets = set(df["dataset"])
    positive_datasets = [
        dataset for dataset in datasets if dataset not in negative_datasets]
    positive_df_ids = [row in positive_datasets for row in df["dataset"]]
    filtered_df = df.loc[positive_df_ids]
    return filtered_df
