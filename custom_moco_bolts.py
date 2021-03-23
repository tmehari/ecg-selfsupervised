import pytorch_lightning as pl
from pl_bolts.models.self_supervised import MocoV2
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim import Adam
import torch
import re
import pdb
from argparse import ArgumentParser
from typing import Union
from warnings import warn
import torch.nn.functional as F
from torch import nn
from pl_bolts.metrics import precision_at_k  # , mean
from clinical_ts.create_logger import create_logger
from models.resnet_simclr import ResNetSimCLR
import re

import time

import yaml
import logging
import pickle
import os
from clinical_ts.simclr_dataset_wrapper import SimCLRDataSetWrapper
from clinical_ts.create_logger import create_logger
import pickle
from pytorch_lightning import Trainer, seed_everything

from torch import nn
from pytorch_lightning.loggers import TensorBoardLogger
from pl_bolts.models.self_supervised.evaluator import Flatten
import pdb
logger = create_logger(__name__)
method="moco"

def _accuracy(zis, zjs, batch_size):
    with torch.no_grad():
        representations = torch.cat([zjs, zis], dim=0)
        similarity_matrix = torch.mm(
            representations, representations.t().contiguous())
        corrected_similarity_matrix = similarity_matrix - \
            torch.eye(2*batch_size).type_as(similarity_matrix)
        pred_similarities, pred_indices = torch.max(
            corrected_similarity_matrix[:batch_size], dim=1)
        correct_indices = torch.arange(batch_size)+batch_size
        correct_preds = (
            pred_indices == correct_indices.type_as(pred_indices)).sum()
    return correct_preds.float()/batch_size


def mean(res, key1, key2=None):
    if key2 is not None:
        return torch.stack([x[key1][key2] for x in res]).mean()
    return torch.stack([x[key1] for x in res if type(x) == dict and key1 in x.keys()]).mean()

# utils


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class CustomMoCo(pl.LightningModule):

    def __init__(self,
                 base_encoder,
                 emb_dim: int = 128,
                 num_negatives: int = 65536,
                 encoder_momentum: float = 0.999,
                 softmax_temperature: float = 0.07,
                 learning_rate: float = 0.03,
                 momentum: float = 0.9,
                 weight_decay: float = 1e-6,
                 datamodule: pl.LightningDataModule = None,
                 data_dir: str = './',
                 batch_size: int = 256,
                 use_mlp: bool = False,
                 num_workers: int = 8,
                 config=None,
                 transformations=None,
                 warmup_epochs=10,
                 *args, **kwargs):

        super(CustomMoCo, self).__init__()
        self.base_encoder = base_encoder
        self.emb_dim = emb_dim
        self.num_negatives = num_negatives
        self.encoder_momentum = encoder_momentum
        self.softmax_temperature = softmax_temperature
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.datamodule = datamodule
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.use_mlp = use_mlp
        self.num_workers = num_workers
        self.warmup_epochs = warmup_epochs
        self.config = config
        self.transformations = transformations
        self.epoch = 0
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q, self.encoder_k = self.init_encoders(base_encoder)

        if use_mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(emb_dim, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # self.warmup_epochs = config["warm_up"]

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

    def setup(self, stage):
        global_batch_size = self.trainer.world_size * self.batch_size
        self.train_iters_per_epoch = self.datamodule.num_samples // global_batch_size


    def configure_optimizers(self):
        # global_batch_size = self.trainer.world_size * self.batch_size
        # self.train_iters_per_epoch = self.datamodule.num_samples // global_batch_size
        # # TRICK 1 (Use lars + filter weights)
        # # exclude certain parameters
        # parameters = self.exclude_from_wt_decay(
        #     self.named_parameters(),
        #     weight_decay=self.weight_decay
        # )

        # optimizer = LARSWrapper(Adam(parameters, lr=self.learning_rate))

        # # Trick 2 (after each step)
        # self.warmup_epochs = self.warmup_epochs * self.train_iters_per_epoch
        # max_epochs = self.trainer.max_epochs * self.train_iters_per_epoch

        # linear_warmup_cosine_decay = LinearWarmupCosineAnnealingLR(
        #     optimizer,
        #     warmup_epochs=self.warmup_epochs,
        #     max_epochs=max_epochs,
        #     warmup_start_lr=0,
        #     eta_min=0
        # )

        # scheduler = {
        #     'scheduler': linear_warmup_cosine_decay,
        #     'interval': 'step',
        #     'frequency': 1
        # }

        # self.scheduler = linear_warmup_cosine_decay

        logger.debug("configure_optimizers")
        optimizer = torch.optim.Adam(self.parameters(
        ), self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config["epochs"], eta_min=0,
                                                                    last_epoch=-1)
        return [optimizer], [self.scheduler]

    def init_encoders(self, base_encoder):
        """
        Override to add your own encoders
        """

        encoder_q = base_encoder()
        encoder_k = base_encoder()

        return encoder_q, encoder_k

    def forward(self, img_q, img_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # ugly fix
        img_q = img_q.type_as(self.encoder_q.features[0][0].weight.data)
        img_k = img_k.type_as(self.encoder_q.features[0][0].weight.data)

        # compute query features
        q = self.encoder_q(img_q)[1]  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            if self.use_ddp or self.use_ddp2:
                img_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)

            k = self.encoder_k(img_k)[1]  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            if self.use_ddp or self.use_ddp2:
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.softmax_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

    def training_step(self, batch, batch_idx):
        (img_1, _), (img_2, _) = batch
        output, target = self(img_q=img_1.float(), img_k=img_2.float())
        loss = F.cross_entropy(output.float(), target.long())
        acc = precision_at_k(output, target, top_k=(1,))[0]

        log = {
            'train_loss': loss,
            'train_acc': acc
        }
        return {'loss': loss, 'log': log, 'progress_bar': log}

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx != 0:
            return {}

        (img_1, _), (img_2, _) = batch

        output, target = self(img_q=img_1, img_k=img_2)
        loss = F.cross_entropy(output, target.long())

        acc = precision_at_k(output, target, top_k=(1,))[0]
        results = {
            'val_loss': loss,
            'val_acc': acc
        }
        return results

    def training_epoch_end(self, outputs):
        train_loss = mean(outputs, 'log', 'train_loss')
        train_acc = mean(outputs, 'log', 'train_acc')

        log = {
            'train/train_loss': train_loss,
            'train/train_acc': train_acc
        }
        return {'train_loss': train_loss, 'log': log, 'progress_bar': log}

    def validation_epoch_end(self, outputs):
        # outputs[0] because we are using multiple datasets!
        val_loss = mean(outputs[0], 'val_loss')
        val_acc = mean(outputs[0], 'val_acc')

        log = {
            'val/val_loss': val_loss,
            'val/val_acc': val_acc
        }
        return {'val_loss': val_loss, 'log': log, 'progress_bar': log}

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
        # import pdb
        # pdb.set_trace()
        self.logger.experiment.add_scalar('cosine_lr_decay', self.scheduler.get_lr()[
            0], global_step=self.epoch)
        self.epoch += 1
        if self.epoch >= 10:
            self.scheduler.step()

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        if self.use_ddp or self.use_ddp2:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        # import pdb
        # pdb.set_trace()
        ptr = int(self.queue_ptr)

        # replace the keys at ptr (dequeue and enqueue)
        remainder = self.queue[:, ptr:ptr + batch_size].shape[1]
        if remainder < batch_size:
            self.queue[:, -remainder:] = keys.T[:, :remainder]
            self.queue[:, :batch_size-remainder] = keys.T[:, remainder:]
            ptr = batch_size-remainder
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.num_negatives  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):  # pragma: no-cover
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):  # pragma: no-cover
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            em = self.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1. - em)

    def type(self):
        return self.encoder_k.features[0][0].weight.type()

    def get_representations(self, x):
        return self.encoder_q.features(x)

    def get_model(self):
        return self.encoder_q
        
    def get_device(self):
        return self.encoder_k.features[0][0].weight.device

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
    parser.add_argument('--checkpoint_path', default="")
    parser.add_argument(
            '--percentage', help='determines how much of the dataset shall be used during the pretraining', type=float, default=1.0)
    parser.add_argument('--lr', type=float, help="learning rate")
    parser.add_argument('--out_dim', type=int, help="output dimension of model")
    parser.add_argument('--filter_cinc', default=False, action="store_true", help="only valid if cinc is selected: filter out the ptb data")
    parser.add_argument('--base_model')
    parser.add_argument('--widen',type=int, help="use wide xresnet1d50")
    parser.add_argument('--run_callbacks', default=False, action="store_true", help="run callbacks which asses linear evaluaton and finetuning metrics during pretraining")

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
    def create_encoder(): return ResNetSimCLR(**config["model"])
    pl_model = CustomMoCo(create_encoder, datamodule=ecg_datamodule, num_negatives=ecg_datamodule.num_samples,
                              emb_dim=config["model"]["out_dim"], config=config, transformations=ecg_datamodule.transformations,
                              batch_size=config["batch_size"], learning_rate=config["lr"], softmax_temperature=config["lr"],
                              warmup_epochs=config["warm_up"], weight_decay=eval(config["weight_decay"]))
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