"""
Adapted from official swav implementation: https://github.com/facebookresearch/swav
"""
import math
import os
import re
from argparse import ArgumentParser
from typing import Callable, Optional
import pdb
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pytorch_lightning.utilities import AMPType
from torch import nn
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim.optimizer import Optimizer

import yaml
import time
import logging
import pickle
# from pl_bolts.models.self_supervised.swav.swav_resnet import resnet18, resnet50
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    imagenet_normalization,
    stl10_normalization,
)
from clinical_ts.simclr_dataset_wrapper import SimCLRDataSetWrapper
from clinical_ts.create_logger import create_logger
from torchvision.models.resnet import Bottleneck, BasicBlock
from online_evaluator import SSLOnlineEvaluator
from ecg_datamodule import ECGDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from models.resnet_simclr import ResNetSimCLR
import torchvision.transforms as transforms

_TORCHVISION_AVAILABLE = True

# import cv2
from typing import List
logger = create_logger(__name__)
method = "swav"
class SwAVTrainDataTransform(object):
    def __init__(
        self,
        normalize=None,
        size_crops: List[int] = [96, 36],
        nmb_crops: List[int] = [2, 4],
        min_scale_crops: List[float] = [0.33, 0.10],
        max_scale_crops: List[float] = [1, 0.33],
        gaussian_blur: bool = True,
        jitter_strength: float = 1.
    ):
        self.jitter_strength = jitter_strength
        self.gaussian_blur = gaussian_blur

        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)

        self.size_crops = size_crops
        self.nmb_crops = nmb_crops
        self.min_scale_crops = min_scale_crops
        self.max_scale_crops = max_scale_crops

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength
        )

        transform = []
        color_transform = [
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ]

        if self.gaussian_blur:
            kernel_size = int(0.1 * self.size_crops[0])
            if kernel_size % 2 == 0:
                kernel_size += 1

            color_transform.append(
                GaussianBlur(kernel_size=kernel_size, p=0.5)
            )

        self.color_transform = transforms.Compose(color_transform)

        if normalize is None:
            self.final_transform = transforms.ToTensor()
        else:
            self.final_transform = transforms.Compose(
                [transforms.ToTensor(), normalize])

        for i in range(len(self.size_crops)):
            random_resized_crop = transforms.RandomResizedCrop(
                self.size_crops[i],
                scale=(self.min_scale_crops[i], self.max_scale_crops[i]),
            )

            transform.extend([transforms.Compose([
                random_resized_crop,
                transforms.RandomHorizontalFlip(p=0.5),
                self.color_transform,
                self.final_transform])
            ] * self.nmb_crops[i])

        self.transform = transform

        # add online train transform of the size of global view
        online_train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.size_crops[0]),
            transforms.RandomHorizontalFlip(),
            self.final_transform
        ])

        self.transform.append(online_train_transform)
        
    def __call__(self, sample):
        multi_crops = list(
            map(lambda transform: transform(sample), self.transform)
        )
        return multi_crops


class SwAVEvalDataTransform(SwAVTrainDataTransform):
    def __init__(
        self,
        normalize=None,
        size_crops: List[int] = [96, 36],
        nmb_crops: List[int] = [2, 4],
        min_scale_crops: List[float] = [0.33, 0.10],
        max_scale_crops: List[float] = [1, 0.33],
        gaussian_blur: bool = True,
        jitter_strength: float = 1.
    ):
        super().__init__(
            normalize=normalize,
            size_crops=size_crops,
            nmb_crops=nmb_crops,
            min_scale_crops=min_scale_crops,
            max_scale_crops=max_scale_crops,
            gaussian_blur=gaussian_blur,
            jitter_strength=jitter_strength
        )

        input_height = self.size_crops[0]  # get global view crop
        test_transform = transforms.Compose([
            transforms.Resize(int(input_height + 0.1 * input_height)),
            transforms.CenterCrop(input_height),
            self.final_transform,
        ])

        # replace last transform to eval transform in self.transform list
        self.transform[-1] = test_transform


class SwAVFinetuneTransform(object):
    def __init__(
        self,
        input_height: int = 224,
        jitter_strength: float = 1.,
        normalize=None,
        eval_transform: bool = False
    ) -> None:

        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.normalize = normalize

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength
        )

        if not eval_transform:
            data_transforms = [
                transforms.RandomResizedCrop(size=self.input_height),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([self.color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2)
            ]
        else:
            data_transforms = [
                transforms.Resize(
                    int(self.input_height + 0.1 * self.input_height)),
                transforms.CenterCrop(self.input_height)
            ]

        if normalize is None:
            final_transform = transforms.ToTensor()
        else:
            final_transform = transforms.Compose(
                [transforms.ToTensor(), normalize])

        data_transforms.append(final_transform)
        self.transform = transforms.Compose(data_transforms)

    def __call__(self, sample):
        return self.transform(sample)


class CustomResNet(nn.Module):
    def __init__(
            self,
            model,
            zero_init_residual=False,
            output_dim=16,
            hidden_mlp=512,
            nmb_prototypes=8,
            eval_mode=False,
            first_conv=True,
            maxpool1=True, 
            l2norm=True
    ):
        super(CustomResNet, self).__init__()
        self.l2norm = l2norm
        self.model = model
        self.features = self.model.features
        self.projection_head = nn.Sequential(
                nn.Linear(512, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_mlp, output_dim),
            )

        # prototype layer
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward_backbone(self, x):
        x = x.type(self.features[0][0].weight.type())
        h = self.features(x)
        h = h.squeeze()
        return h

    def forward_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in inputs]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            _out = torch.cat(inputs[start_idx: end_idx])

            if 'cuda' in str(self.features[0][0].weight.device):
                _out = self.forward_backbone(_out.cuda(non_blocking=True))
            else:
                _out = self.forward_backbone(_out)

            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        return self.forward_head(output)


class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i),
                            nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out


class CustomSwAV(pl.LightningModule):
    def __init__(
        self,
        model,
        gpus: int,
        num_samples: int,
        batch_size: int,
        config=None,
        transformations=None,
        nodes: int = 1,
        arch: str = 'resnet50',
        hidden_mlp: int = 2048,
        feat_dim: int = 128,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        nmb_prototypes: int = 3000,
        freeze_prototypes_epochs: int = 1,
        temperature: float = 0.1,
        sinkhorn_iterations: int = 3,
        # queue_length: int = 512,  # must be divisible by total batch-size
        queue_path: str = "queue",
        epoch_queue_starts: int = 15,
        crops_for_assign: list = [0, 1],
        nmb_crops: list = [2, 6],
        first_conv: bool = True,
        maxpool1: bool = True,
        optimizer: str = 'adam',
        lars_wrapper: bool = False,
        exclude_bn_bias: bool = False,
        start_lr: float = 0.,
        learning_rate: float = 1e-3,
        final_lr: float = 0.,
        weight_decay: float = 1e-6,
        epsilon: float = 0.05,
        **kwargs
    ):
        """
        Args:
            gpus: number of gpus per node used in training, passed to SwAV module
                to manage the queue and select distributed sinkhorn
            nodes: number of nodes to train on
            num_samples: number of image samples used for training
            batch_size: batch size per GPU in ddp
            dataset: dataset being used for train/val
            arch: encoder architecture used for pre-training
            hidden_mlp: hidden layer of non-linear projection head, set to 0
                to use a linear projection head
            feat_dim: output dim of the projection head
            warmup_epochs: apply linear warmup for this many epochs
            max_epochs: epoch count for pre-training
            nmb_prototypes: count of prototype vectors
            freeze_prototypes_epochs: epoch till which gradients of prototype layer
                are frozen
            temperature: loss temperature
            sinkhorn_iterations: iterations for sinkhorn normalization
            queue_length: set queue when batch size is small,
                must be divisible by total batch-size (i.e. total_gpus * batch_size),
                set to 0 to remove the queue
            queue_path: folder within the logs directory
            epoch_queue_starts: start uing the queue after this epoch
            crops_for_assign: list of crop ids for computing assignment
            nmb_crops: number of global and local crops, ex: [2, 6]
            first_conv: keep first conv same as the original resnet architecture,
                if set to false it is replace by a kernel 3, stride 1 conv (cifar-10)
            maxpool1: keep first maxpool layer same as the original resnet architecture,
                if set to false, first maxpool is turned off (cifar10, maybe stl10)
            optimizer: optimizer to use
            lars_wrapper: use LARS wrapper over the optimizer
            exclude_bn_bias: exclude batchnorm and bias layers from weight decay in optimizers
            start_lr: starting lr for linear warmup
            learning_rate: learning rate
            final_lr: float = final learning rate for cosine weight decay
            weight_decay: weight decay for optimizer
            epsilon: epsilon val for swav assignments
        """
        super().__init__()
        # self.save_hyperparameters()

        self.epoch = 0
        self.config = config
        self.transformations = transformations
        self.gpus = gpus
        self.nodes = nodes
        self.arch = arch
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.queue_length = 8*batch_size

        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim
        self.nmb_prototypes = nmb_prototypes
        self.freeze_prototypes_epochs = freeze_prototypes_epochs
        self.sinkhorn_iterations = sinkhorn_iterations

        #self.queue_length = queue_length
        self.queue_path = queue_path
        self.epoch_queue_starts = epoch_queue_starts
        self.crops_for_assign = crops_for_assign
        self.nmb_crops = nmb_crops

        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        self.optim = optimizer
        self.lars_wrapper = lars_wrapper
        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.temperature = temperature

        self.start_lr = start_lr
        self.final_lr = final_lr
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = config["epochs"]

        if self.gpus * self.nodes > 1:
            self.get_assignments = self.distributed_sinkhorn
        else:
            self.get_assignments = self.sinkhorn

        
        
        # compute iters per epoch
        global_batch_size = self.nodes * self.gpus * \
            self.batch_size if self.gpus > 0 else self.batch_size
        self.train_iters_per_epoch = (self.num_samples // global_batch_size)+1

        # define LR schedule
        warmup_lr_schedule = np.linspace(
            self.start_lr, self.learning_rate, self.train_iters_per_epoch * self.warmup_epochs
        )
        iters = np.arange(self.train_iters_per_epoch *
                          (self.max_epochs - self.warmup_epochs))
        cosine_lr_schedule = np.array([self.final_lr + 0.5 * (self.learning_rate - self.final_lr) * (
            1 + math.cos(math.pi * t / (self.train_iters_per_epoch *
                                        (self.max_epochs - self.warmup_epochs)))
        ) for t in iters])

        self.lr_schedule = np.concatenate(
            (warmup_lr_schedule, cosine_lr_schedule))
        self.queue = None   
        self.model = self.init_model(model)
        self.softmax = nn.Softmax(dim=1)
        

    def setup(self, stage):
        queue_folder = os.path.join(self.config["log_dir"], self.queue_path)
        if not os.path.exists(queue_folder):
            os.makedirs(queue_folder)

        self.queue_path = os.path.join(
            queue_folder,
            "queue" + str(self.trainer.global_rank) + ".pth"
        )

        if os.path.isfile(self.queue_path):
            self.queue = torch.load(self.queue_path)["queue"]
        
    def init_model(self, model):
        return CustomResNet(model, hidden_mlp=self.hidden_mlp,
            output_dim=self.feat_dim,
            nmb_prototypes=self.nmb_prototypes,
            first_conv=self.first_conv,
            maxpool1=self.maxpool1)

    def forward(self, x):
        # pass single batch from the resnet backbone
        return self.model.forward_backbone(x)
    
    def on_train_start(self):
        # # log configuration
        # config_str = re.sub(r"[,\}\{]", "<br/>", str(self.config))
        # config_str = re.sub(r"[\[\]\']", "", config_str)
        # transformation_str = re.sub(r"[\}]", "<br/>", str(["<br>" + str(
        #     t) + ":<br/>" + str(t.get_params()) for t in self.transformations]))
        # transformation_str = re.sub(r"[,\"\{\'\[\]]", "", transformation_str)
        # self.logger.experiment.add_text(
        #     "configuration", str(config_str), global_step=0)
        # self.logger.experiment.add_text("transformations", str(
        #     transformation_str), global_step=0)
        self.epoch = 0

    def on_train_epoch_start(self):
        if self.queue_length > 0:
            if self.trainer.current_epoch >= self.epoch_queue_starts and self.queue is None:
                self.queue = torch.zeros(
                    len(self.crops_for_assign),
                    self.queue_length // self.gpus,  # change to nodes * gpus once multi-node
                    self.feat_dim,
                )

                if self.gpus > 0:
                    self.queue = self.queue.cuda()

        self.use_the_queue = False

    def on_train_epoch_end(self, outputs) -> None:
        if self.queue is not None:
            torch.save({"queue": self.queue}, self.queue_path)

    def on_epoch_end(self):
        self.epoch += 1

    def on_after_backward(self):
        if self.current_epoch < self.freeze_prototypes_epochs:
            for name, p in self.model.named_parameters():
                if "prototypes" in name:
                    p.grad = None

    def shared_step(self, batch):
        # if self.dataset == 'stl10':
        #     unlabeled_batch = batch[0]
        #     batch = unlabeled_batch
        
        
        inputs, y = batch
        # remove online train/eval transforms at this point
        inputs = inputs[:-1]

        # 1. normalize the prototypes
        with torch.no_grad():
            w = self.model.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.model.prototypes.weight.copy_(w)

        # 2. multi-res forward passes
        embedding, output = self.model(inputs)
        embedding = embedding.detach()
        bs = inputs[0].size(0)

        # 3. swav loss computation
        loss = 0
        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)]

                # 4. time to use the queue
                if self.queue is not None:
                    if self.use_the_queue or not torch.all(self.queue[i, -1, :] == 0):
                        self.use_the_queue = True
                        out = torch.cat((torch.mm(
                            self.queue[i],
                            self.model.prototypes.weight.t()
                        ), out))
                    # fill the queue
                    self.queue[i, bs:] = self.queue[i, :-bs].clone()
                    self.queue[i, :bs] = embedding[crop_id *
                                                   bs: (crop_id + 1) * bs]

                # 5. get assignments
                q = torch.exp(out / self.epsilon).t()
                q = self.get_assignments(q, self.sinkhorn_iterations)[-bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(self.nmb_crops-1)), crop_id):
                p = self.softmax(
                    output[bs * v: bs * (v + 1)] / self.temperature)
                loss_value = q * torch.log(p)
                subloss -= torch.mean(torch.sum(loss_value, dim=1))
            loss += subloss / (np.sum(self.nmb_crops) - 1)
        loss /= len(self.crops_for_assign)

        return loss

    def training_step(self, batch, batch_idx):
        
        loss = self.shared_step(batch)

        # self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        
        if dataloader_idx != 0:
            return {}
        loss = self.shared_step(batch)

        # self.log('val_loss', loss, on_step=False, on_epoch=True)
        results = {
            'val_loss': loss,
        }
        return results
    
    def validation_epoch_end(self, outputs):
        # outputs[0] because we are using multiple datasets!
        val_loss = mean(outputs[0], 'val_loss')

        log = {
            'val/val_loss': val_loss,
        }
        return {'val_loss': val_loss, 'log': log, 'progress_bar': log}

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=['bias', 'bn']):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {'params': params, 'weight_decay': weight_decay},
            {'params': excluded_params, 'weight_decay': 0.}
        ]

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(
                self.named_parameters(),
                weight_decay=self.weight_decay
            )
        else:
            params = self.parameters()

        if self.optim == 'sgd':
            optimizer = torch.optim.SGD(
                params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        elif self.optim == 'adam':
            optimizer = torch.optim.Adam(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )

        if self.lars_wrapper:
            optimizer = LARSWrapper(
                optimizer,
                eta=0.001,  # trust coefficient
                clip=False
            )

        return optimizer
    
    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer: Optimizer = None,
        optimizer_idx: int = None,
        optimizer_closure: Optional[Callable] = None,
        on_tpu: bool = None,
        using_native_amp: bool = None,
        using_lbfgs: bool = None,
    ) -> None:
        # warm-up + decay schedule placed here since LARSWrapper is not optimizer class
        # adjust LR of optim contained within LARSWrapper
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.lr_schedule[self.trainer.global_step]

        # from lightning
        if not isinstance(optimizer, LightningOptimizer):
            # wraps into LightingOptimizer only for running step
            optimizer = LightningOptimizer.to_lightning_optimizer(optimizer, self.trainer)
        optimizer.step(closure=optimizer_closure)
        
    def sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            K, B = Q.shape

            if self.gpus > 0:
                u = torch.zeros(K).cuda()
                r = torch.ones(K).cuda() / K
                c = torch.ones(B).cuda() / B
            else:
                u = torch.zeros(K)
                r = torch.ones(K) / K
                c = torch.ones(B) / B

            for _ in range(nmb_iters):
                u = torch.sum(Q, dim=1)

                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def distributed_sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            dist.all_reduce(sum_Q)
            Q /= sum_Q

            if self.gpus > 0:
                u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
                r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
                c = torch.ones(Q.shape[1]).cuda(
                    non_blocking=True) / (self.gpus * Q.shape[1])
            else:
                u = torch.zeros(Q.shape[0])
                r = torch.ones(Q.shape[0]) / Q.shape[0]
                c = torch.ones(Q.shape[1]) / (self.gpus * Q.shape[1])

            curr_sum = torch.sum(Q, dim=1)
            dist.all_reduce(curr_sum)

            for it in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
                dist.all_reduce(curr_sum)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def type(self):
        return self.model.features[0][0].weight.type()

    def get_representations(self, x):
        return self.model.features(x)

    def get_model(self):
        return self.model.model
        
    def get_device(self):
        return self.model.features[0][0].weight.device

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # model params
        parser.add_argument("--arch", default="resnet50",
                            type=str, help="convnet architecture")
        # specify flags to store false
        parser.add_argument("--first_conv", action='store_false')
        parser.add_argument("--maxpool1", action='store_false')
        parser.add_argument("--hidden_mlp", default=2048, type=int,
                            help="hidden layer dimension in projection head")
        parser.add_argument("--feat_dim", default=128,
                            type=int, help="feature dimension")
        parser.add_argument("--online_ft", action='store_true')
        parser.add_argument("--fp32", action='store_true')

        # transform params
        parser.add_argument("--gaussian_blur",
                            action="store_true", help="add gaussian blur")
        parser.add_argument("--jitter_strength", type=float,
                            default=1.0, help="jitter strength")
        parser.add_argument("--dataset", type=str,
                            default="stl10", help="stl10, cifar10")
        parser.add_argument("--data_dir", type=str,
                            default=".", help="path to download data")
        parser.add_argument("--queue_path", type=str,
                            default="queue", help="path for queue")

        parser.add_argument("--nmb_crops", type=int, default=[2, 4], nargs="+",
                            help="list of number of crops (example: [2, 6])")
        parser.add_argument("--size_crops", type=int, default=[96, 36], nargs="+",
                            help="crops resolutions (example: [224, 96])")
        parser.add_argument("--min_scale_crops", type=float, default=[0.33, 0.10], nargs="+",
                            help="argument in RandomResizedCrop (example: [0.14, 0.05])")
        parser.add_argument("--max_scale_crops", type=float, default=[1, 0.33], nargs="+",
                            help="argument in RandomResizedCrop (example: [1., 0.14])")

        # training params
        parser.add_argument("--fast_dev_run", action='store_true')
        parser.add_argument("--nodes", default=1, type=int,
                            help="number of nodes for training")
        parser.add_argument("--gpus", default=1, type=int,
                            help="number of gpus to train on")
        parser.add_argument("--num_workers", default=8,
                            type=int, help="num of workers per GPU")
        parser.add_argument("--optimizer", default="adam",
                            type=str, help="choose between adam/sgd")
        parser.add_argument("--lars_wrapper", action='store_true',
                            help="apple lars wrapper over optimizer used")
        parser.add_argument('--exclude_bn_bias', action='store_true',
                            help="exclude bn/bias from weight decay")
        parser.add_argument("--max_epochs", default=100,
                            type=int, help="number of total epochs to run")
        parser.add_argument("--max_steps", default=-1,
                            type=int, help="max steps")
        parser.add_argument("--warmup_epochs", default=10,
                            type=int, help="number of warmup epochs")
        parser.add_argument("--batch_size", default=128,
                            type=int, help="batch size per gpu")

        parser.add_argument("--weight_decay", default=1e-6,
                            type=float, help="weight decay")
        parser.add_argument("--learning_rate", default=1e-3,
                            type=float, help="base learning rate")
        parser.add_argument("--start_lr", default=0, type=float,
                            help="initial warmup learning rate")
        parser.add_argument("--final_lr", type=float,
                            default=1e-6, help="final learning rate")

        # swav params
        parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                            help="list of crops id used for computing assignments")
        parser.add_argument("--temperature", default=0.1, type=float,
                            help="temperature parameter in training loss")
        parser.add_argument("--epsilon", default=0.05, type=float,
                            help="regularization parameter for Sinkhorn-Knopp algorithm")
        parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                            help="number of iterations in Sinkhorn-Knopp algorithm")
        parser.add_argument("--nmb_prototypes", default=512,
                            type=int, help="number of prototypes")
        parser.add_argument("--queue_length", type=int, default=0,
                            help="length of the queue (0 for no queue); must be divisible by total batch size")
        parser.add_argument("--epoch_queue_starts", type=int, default=15,
                            help="from this epoch, we start using a queue")
        parser.add_argument("--freeze_prototypes_epochs", default=1, type=int,
                            help="freeze the prototypes during this many epochs from the start")

        return parser


def mean(res, key1, key2=None):
    if key2 is not None:
        return torch.stack([x[key1][key2] for x in res]).mean()
    return torch.stack([x[key1] for x in res if type(x) == dict and key1 in x.keys()]).mean()

def parse_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('-t', '--trafos', nargs='+', help='add transformation to data augmentation pipeline',
                        default=["GaussianNoise", "ChannelResize", "RandomResizedCrop"])
    # GaussianNoise
    parser.add_argument(
            '--gaussian_scale', help='std param for gaussian noise transformation', default=0.005, type=float)
    # RandomResizedCrop
    parser.add_argument('--rr_crop_ratio_range',
                            help='ratio range for random resized crop transformation', default=[0.5, 1.0], type=float)
    parser.add_argument(
            '--output_size', help='output size for random resized crop transformation', default=250, type=int)
    # DynamicTimeWarp
    parser.add_argument(
            '--warps', help='number of warps for dynamic time warp transformation', default=3, type=int)
    parser.add_argument(
            '--radius', help='radius of warps of dynamic time warp transformation', default=10, type=int)
    # TimeWarp
    parser.add_argument(
            '--epsilon', help='epsilon param for time warp', default=10, type=float)
    # ChannelResize
    parser.add_argument('--magnitude_range', nargs='+',
                            help='range for scale param for ChannelResize transformation', default=[0.5, 2], type=float)
    # Downsample
    parser.add_argument(
            '--downsample_ratio', help='downsample ratio for Downsample transformation', default=0.2, type=float)
    # TimeOut
    parser.add_argument('--to_crop_ratio_range', nargs='+',
                            help='ratio range for timeout transformation', default=[0.2, 0.4], type=float)
    # resume training
    parser.add_argument('--resume', action='store_true')
    parser.add_argument(
            '--gpus', help='number of gpus to use; use cpu if gpu=0', type=int, default=1)
    parser.add_argument(
            '--num_nodes', default=1, help='number of cluster nodes', type=int)
    parser.add_argument(
            '--distributed_backend', help='sets backend type')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--warm_up', default=1, type=int)
    parser.add_argument('--precision', type=int)
    parser.add_argument('--datasets', dest="target_folders",
                            nargs='+', help='used datasets for pretraining')
    parser.add_argument('--log_dir', default="./experiment_logs")
    parser.add_argument(
            '--percentage', help='determines how much of the dataset shall be used during the pretraining', type=float, default=1.0)
    parser.add_argument('--lr', type=float, help="learning rate")
    parser.add_argument('--out_dim', type=int, help="output dimension of model")
    parser.add_argument('--filter_cinc', default=False, action="store_true", help="only valid if cinc is selected: filter out the ptb data")
    parser.add_argument('--base_model')
    parser.add_argument('--widen',type=int, help="use wide xresnet1d50")
    parser.add_argument('--run_callbacks', default=False, action="store_true", help="run callbacks which asses linear evaluaton and finetuning metrics during pretraining")

    parser.add_argument('--checkpoint_path', default="")
    return parser

def init_logger(config):
    level = logging.INFO

    if config['debug']:
        level = logging.DEBUG

    # remove all handlers to change basic configuration
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if not os.path.isdir(config['log_dir']):
        os.mkdir(config['log_dir'])
    logging.basicConfig(filename=os.path.join(config['log_dir'], 'info.log'), level=level,
                        format='%(asctime)s %(name)s:%(lineno)s %(levelname)s:  %(message)s  ')
    return logging.getLogger(__name__)

def pretrain_routine(args):
    t_params = {"gaussian_scale": args.gaussian_scale, "rr_crop_ratio_range": args.rr_crop_ratio_range, "output_size": args.output_size, "warps": args.warps, "radius": args.radius,
                "epsilon": args.epsilon, "magnitude_range": args.magnitude_range, "downsample_ratio": args.downsample_ratio, "to_crop_ratio_range": args.to_crop_ratio_range,
                "bw_cmax":0.1, "em_cmax":0.5, "pl_cmax":0.2, "bs_cmax":1}
    transformations = args.trafos
    checkpoint_config = os.path.join("checkpoints", "bolts_config.yaml")
    config_file = checkpoint_config if args.resume and os.path.isfile(
        checkpoint_config) else "bolts_config.yaml"
    config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
    args_dict = vars(args)
    for key in set(config.keys()).union(set(args_dict.keys())):
        config[key] = config[key] if (key not in args_dict.keys() or key in args_dict.keys(
        ) and key in config.keys() and args_dict[key] is None) else args_dict[key]
    if args.target_folders is not None:
        config["dataset"]["target_folders"] = args.target_folders
    config["dataset"]["percentage"] = args.percentage if args.percentage is not None else config["dataset"]["percentage"]
    config["dataset"]["filter_cinc"] = args.filter_cinc if args.filter_cinc is not None else config["dataset"]["filter_cinc"]
    config["model"]["base_model"] = args.base_model if args.base_model is not None else config["model"]["base_model"]
    config["model"]["widen"] = args.widen if args.widen is not None else config["model"]["widen"]
    config["dataset"]["swav"] = True
    config["dataset"]["nmb_crops"] = 7
    config["eval_dataset"]["swav"] = True
    config["eval_dataset"]["nmb_crops"] = 7
    if args.out_dim is not None:
        config["model"]["out_dim"] = args.out_dim
    init_logger(config)
    dataset = SimCLRDataSetWrapper(
        config['batch_size'], **config['dataset'], transformations=transformations, t_params=t_params)
    for i, t in enumerate(dataset.transformations):
        logger.info(str(i) + ". Transformation: " +
                    str(t) + ": " + str(t.get_params()))
    date = time.asctime()
    label_to_num_classes = {"label_all": 71, "label_diag": 44, "label_form": 19,
                            "label_rhythm": 12, "label_diag_subclass": 23, "label_diag_superclass": 5}
    ptb_num_classes = label_to_num_classes[config["eval_dataset"]
                                           ["ptb_xl_label"]]
    abr = {"Transpose": "Tr", "TimeOut": "TO", "DynamicTimeWarp": "DTW", "RandomResizedCrop": "RRC", "ChannelResize": "ChR", "GaussianNoise": "GN",
           "TimeWarp": "TW", "ToTensor": "TT", "GaussianBlur": "GB", "BaselineWander": "BlW", "PowerlineNoise": "PlN", "EMNoise": "EM", "BaselineShift": "BlS"}
    trs = re.sub(r"[,'\]\[]", "", str([abr[str(tr)] if abr[str(tr)] not in [
                 "TT", "Tr"] else '' for tr in dataset.transformations]))
    name = str(date) + "_" + method + "_" + str(
        time.time_ns())[-3:] + "_" + trs[1:]
    tb_logger = TensorBoardLogger(args.log_dir, name=name, version='')
    config["log_dir"] = os.path.join(args.log_dir, name)
    print(config)
    return config, dataset, date, transformations, t_params, ptb_num_classes, tb_logger

def aftertrain_routine(config, args, trainer, pl_model, datamodule, callbacks):
    scores = {}
    for ca in callbacks:
        if isinstance(ca, SSLOnlineEvaluator):
            scores[str(ca)] = {"macro": ca.best_macro}

    results = {"config": config, "trafos": args.trafos, "scores": scores}

    with open(os.path.join(config["log_dir"], "results.pkl"), 'wb') as handle:
        pickle.dump(results, handle)

    trainer.save_checkpoint(os.path.join(config["log_dir"], "checkpoints", "model.ckpt"))
    with open(os.path.join(config["log_dir"], "config.txt"), "w") as text_file:
        print(config, file=text_file)

def cli_main():
    from pytorch_lightning import Trainer
    from online_evaluator import SSLOnlineEvaluator
    from ecg_datamodule import ECGDataModule
    from clinical_ts.create_logger import create_logger
    from os.path import exists
    
    parser = ArgumentParser()
    parser = parse_args(parser)
    logger.info("parse arguments")
    args = parser.parse_args()
    config, dataset, date, transformations, t_params, ptb_num_classes, tb_logger = pretrain_routine(args)

    # data
    ecg_datamodule = ECGDataModule(config, transformations, t_params)

    callbacks = []
    if args.run_callbacks:
            # callback for online linear evaluation/fine-tuning
        linear_evaluator = SSLOnlineEvaluator(drop_p=0,
                                          z_dim=512, num_classes=ptb_num_classes, hidden_dim=None, lin_eval_epochs=config["eval_epochs"], eval_every=config["eval_every"], mode="linear_evaluation", verbose=False)

        fine_tuner = SSLOnlineEvaluator(drop_p=0,
                                          z_dim=512, num_classes=ptb_num_classes, hidden_dim=None, lin_eval_epochs=config["eval_epochs"], eval_every=config["eval_every"], mode="fine_tuning", verbose=False)
   
        callbacks.append(linear_evaluator)
        callbacks.append(fine_tuner)

    # configure trainer
    trainer = Trainer(logger=tb_logger, max_epochs=config["epochs"], gpus=args.gpus,
                      distributed_backend=args.distributed_backend, auto_lr_find=False, num_nodes=args.num_nodes, precision=config["precision"], callbacks=callbacks)

    # pytorch lightning module
    model = ResNetSimCLR(**config["model"])
    pl_model = CustomSwAV(model,  config["gpus"], ecg_datamodule.num_samples, config["batch_size"], config=config,
                              transformations=ecg_datamodule.transformations, nmb_crops=config["dataset"]["nmb_crops"])
    # load checkpoint
    if args.checkpoint_path != "":
        if exists(args.checkpoint_path):
            logger.info("Retrieve checkpoint from " + args.checkpoint_path)
            pl_model.load_from_checkpoint(args.checkpoint_path)
        else:
            raise("checkpoint does not exist")

    # start training
    trainer.fit(pl_model, ecg_datamodule)

    aftertrain_routine(config, args, trainer, pl_model, ecg_datamodule, callbacks)

if __name__ == "__main__":  
    cli_main()