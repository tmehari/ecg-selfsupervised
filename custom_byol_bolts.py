import math
from argparse import ArgumentParser
from copy import deepcopy
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch.optim import Adam

from pl_bolts.models.self_supervised import BYOL
# from pl_bolts.callbacks.self_supervised import BYOLMAWeightUpdate
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from models.resnet_simclr import ResNetSimCLR
import re

import time

import yaml
import logging
import os
from clinical_ts.simclr_dataset_wrapper import SimCLRDataSetWrapper
from clinical_ts.create_logger import create_logger
import pickle
from pytorch_lightning import Trainer, seed_everything

from torch import nn
from torch.nn import functional as F
from online_evaluator import SSLOnlineEvaluator
from ecg_datamodule import ECGDataModule
from pytorch_lightning.loggers import TensorBoardLogger
import pdb

logger = create_logger(__name__)
method="byol"
def mean(res, key1, key2=None):
    if key2 is not None:
        return torch.stack([x[key1][key2] for x in res]).mean()
    return torch.stack([x[key1] for x in res if type(x) == dict and key1 in x.keys()]).mean()

class MLP(nn.Module):
    def __init__(self, input_dim=512, hidden_size=4096, output_dim=256):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim, bias=True))

    def forward(self, x):
        x = self.model(x)
        return x


class SiameseArm(nn.Module):
    def __init__(self, encoder=None, out_dim=128, hidden_size=512, projector_dim=512):
        super().__init__()

        if encoder is None:
            encoder = torchvision_ssl_encoder('resnet50')
        # Encoder
        self.encoder = encoder
        # Pooler
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        # Projector
        projector_dim = encoder.l1.in_features
        self.projector = MLP(
            input_dim=projector_dim, hidden_size=hidden_size, output_dim=out_dim)
        # Predictor
        self.predictor = MLP(
            input_dim=out_dim, hidden_size=hidden_size, output_dim=out_dim)

    def forward(self, x):
        y = self.encoder(x)[0]
        y = y.view(y.size(0), -1)
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h


class BYOLMAWeightUpdate(pl.Callback):
    def __init__(self, initial_tau=0.996):
        """
        Args:
            initial_tau: starting tau. Auto-updates with every training step
        """
        super().__init__()
        self.initial_tau = initial_tau
        self.current_tau = initial_tau

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # get networks
        online_net = pl_module.online_network
        target_net = pl_module.target_network

        # update weights
        self.update_weights(online_net, target_net)

        # update tau after
        self.current_tau = self.update_tau(pl_module, trainer)

    def update_tau(self, pl_module, trainer):
        max_steps = len(trainer.train_dataloader) * trainer.max_epochs
        tau = 1 - (1 - self.initial_tau) * (math.cos(math.pi *
                                                     pl_module.global_step / max_steps) + 1) / 2
        return tau

    def update_weights(self, online_net, target_net):
        # apply MA weight update
        for (name, online_p), (_, target_p) in zip(online_net.named_parameters(), target_net.named_parameters()):
            if 'weight' in name:
                target_p.data = self.current_tau * target_p.data + \
                    (1 - self.current_tau) * online_p.data


class CustomBYOL(pl.LightningModule):
    def __init__(self,
                 num_classes=5,
                 learning_rate: float = 0.2,
                 weight_decay: float = 1.5e-6,
                 input_height: int = 32,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 warmup_epochs: int = 10,
                 max_epochs: int = 1000,
                 config=None,
                 transformations=None,
                 **kwargs):
        """
        Args:
            datamodule: The datamodule
            learning_rate: the learning rate
            weight_decay: optimizer weight decay
            input_height: image input height
            batch_size: the batch size
            num_workers: number of workers
            warmup_epochs: num of epochs for scheduler warm up
            max_epochs: max epochs for scheduler
        """
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.transformations = transformations
        self.online_network = SiameseArm(
            encoder=self.init_model(), out_dim=config["model"]["out_dim"])
        self.target_network = deepcopy(self.online_network)
        self.weight_callback = BYOLMAWeightUpdate()
        self.log_dict = {}
        self.epoch = 0
        # self.model_device = self.online_network.encoder.features[0][0].weight.device

    def init_model(self):
        model = ResNetSimCLR(**self.config["model"])
        # return model.features
        return model

    # def on_train_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        # Add callback for user automatically since it's key to BYOL weight update
        self.weight_callback.on_train_batch_end(
            self.trainer, self, outputs, batch, batch_idx, 0)

    def forward(self, x):
        y, _, _ = self.online_network(x)
        return y

    def cosine_similarity(self, a, b):
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        sim = (a * b).sum(-1).mean()
        return sim

    def shared_step(self, batch, batch_idx):
        # (img_1, img_2), y = batch
        (img_1, y1), (img_2, y2) = batch

        img_1 = self.to_device(img_1)
        img_2 = self.to_device(img_2)

        # Image 1 to image 2 loss
        y1, z1, h1 = self.online_network(img_1)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(img_2)
        loss_a = - 2 * self.cosine_similarity(h1, z2)

        # Image 2 to image 1 loss
        y1, z1, h1 = self.online_network(img_2)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(img_1)
        # L2 normalize
        loss_b = - 2 * self.cosine_similarity(h1, z2)

        # Final loss
        total_loss = loss_a + loss_b

        return loss_a, loss_b, total_loss

    def training_step(self, batch, batch_idx):
        loss_a, loss_b, total_loss = self.shared_step(batch, batch_idx)

        # log results
        # result = pl.TrainResult(minimize=total_loss)
        # result.log('train_loss/1_2_loss', loss_a, on_epoch=True)
        # result.log('train_loss/2_1_loss', loss_b, on_epoch=True)
        # result.log('train_loss/total_loss', total_loss, on_epoch=True)

        # # log results
        # self.log_dict({'1_2_loss': loss_a, '2_1_loss': loss_b,
        #               'train_loss': total_loss})

        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx != 0:
            return {}

        loss_a, loss_b, total_loss = self.shared_step(batch, batch_idx)

        # # log results
        # result = pl.EvalResult()
        # result.log('val_loss/1_2_loss', loss_a, on_epoch=True)
        # result.log('val_loss/2_1_loss', loss_b, on_epoch=True)
        # result.log('val_loss/total_loss', total_loss, on_epoch=True)

        # self.log_dict({'1_2_loss': loss_a, '2_1_loss': loss_b,
        #               'train_loss': total_loss})
        results = {
            'val_loss': total_loss,
            'val_1_2_loss' : loss_a,
            'val_2_1_loss': loss_b
        }
        return results
    
    def validation_epoch_end(self, outputs):
        # outputs[0] because we are using multiple datasets!
        val_loss = mean(outputs[0], 'val_loss')
        loss_a = mean(outputs[0], 'val_1_2_loss')
        loss_b = mean(outputs[0], 'val_2_1_loss')

        log = {
            'val_loss': val_loss,
            'val_1_2_loss' : loss_a,
            'val_2_1_loss': loss_b
        }

        return {'val_loss': val_loss, 'log': log, 'progress_bar': log}
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate,
                         weight_decay=self.hparams.weight_decay)
        # optimizer = LARSWrapper(optimizer)
        optimizer = optimizer
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs
        )
        return [optimizer], [scheduler]

    def on_train_start(self):
        # log configuration
        config_str = re.sub(r"[,\}\{]", "<br/>", str(self.config))
        config_str = re.sub(r"[\[\]\']", "", config_str)
        transformation_str = re.sub(r"[\}]", "<br/>", str(["<br>" + str(
            t) + ":<br/>" + str(t.get_params()) for t in self.transformations]))
        transformation_str = re.sub(r"[,\"\{\'\[\]]", "", transformation_str)
        self.logger.experiment.add_text(
            "configuration", str(config_str), global_step=0)
        self.logger.experiment.add_text("transformations", str(
            transformation_str), global_step=0)
        self.epoch = 0

    def on_epoch_end(self):
        self.epoch += 1

    def get_representations(self, x):
        return self.online_network(x)[0]

    def get_model(self):
        return self.online_network.encoder

    def get_device(self):
        return self.online_network.encoder.features[0][0].weight.device

    def to_device(self, x):
        return x.type(self.type()).to(self.get_device())

    def type(self):
        return self.online_network.encoder.features[0][0].weight.type()

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
    pl_model = CustomBYOL(5, learning_rate=config["lr"], weight_decay=eval(config["weight_decay"]),
                              warm_up_epochs=config["warm_up"], max_epochs=config[
                                  "epochs"], num_workers=config["dataset"]["num_workers"],
                              batch_size=config["batch_size"], config=config, transformations=ecg_datamodule.transformations)


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