###############
#generic
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import torch.nn.functional as F

import torchvision
import os
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import copy

#################
#specific
from clinical_ts.timeseries_utils import *
from clinical_ts.ecg_utils import *

from functools import partial
from pathlib import Path
import pandas as pd
import numpy as np

from clinical_ts.xresnet1d import xresnet1d50,xresnet1d101
from clinical_ts.basic_conv1d import weight_init
from clinical_ts.eval_utils_cafa import eval_scores, eval_scores_bootstrap
from clinical_ts.cpc import *

def _freeze_bn_stats(model, freeze=True):
    for m in model.modules():
        if(isinstance(m,nn.BatchNorm1d)):
            if(freeze):
                m.eval()
            else:
                m.train()
                
def sanity_check(model, state_dict_pre):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading state dict for sanity check")
    state_dict = model.state_dict()

    for k in list(state_dict.keys()):
        print(k)
        # only ignore fc layer
        if 'head.1.weight' in k or 'head.1.bias' in k:
            continue


        assert ((state_dict[k].cpu() == state_dict_pre[k].cpu()).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")

class LightningCPC(pl.LightningModule):

    def __init__(self, hparams):
        super(LightningCPC, self).__init__()
        
        self.hparams = hparams
        self.lr = self.hparams.lr
        
        #these coincide with the adapted wav2vec2 params
        if(self.hparams.fc_encoder):
            strides=[1]*4 
            kss = [1]*4 
            features = [512]*4
        else: #strided conv encoder
            strides=[2,2,2,2] #original wav2vec2 [5,2,2,2,2,2] original cpc [5,4,2,2,2]
            kss = [10,4,4,4] #original wav2vec2 [10,3,3,3,3,2] original cpc [18,8,4,4,4]
            features = [512]*4 #wav2vec2 [512]*6 original cpc [512]*5
        
        if(self.hparams.finetune):
            self.criterion = F.cross_entropy if self.hparams.finetune_dataset == "thew" else F.binary_cross_entropy_with_logits
            if(self.hparams.finetune_dataset == "thew"):
                num_classes = 5
            elif(self.hparams.finetune_dataset == "ptbxl_super"):
                num_classes = 5
            if(self.hparams.finetune_dataset == "ptbxl_all"):
                num_classes = 71
        else:
            num_classes = None

        self.model_cpc = CPCModel(input_channels=self.hparams.input_channels, strides=strides,kss=kss,features=features,n_hidden=self.hparams.n_hidden,n_layers=self.hparams.n_layers,mlp=self.hparams.mlp,lstm=not(self.hparams.gru),bias_proj=self.hparams.bias,num_classes=num_classes,skip_encoder=self.hparams.skip_encoder,bn_encoder=not(self.hparams.no_bn_encoder),lin_ftrs_head=[] if self.hparams.linear_eval else eval(self.hparams.lin_ftrs_head),ps_head=0 if self.hparams.linear_eval else self.hparams.dropout_head,bn_head=False if self.hparams.linear_eval else not(self.hparams.no_bn_head))
        
        target_fs=100
        if(not(self.hparams.finetune)):
            print("CPC pretraining:\ndownsampling factor:",self.model_cpc.encoder_downsampling_factor,"\nchunk length(s)",self.model_cpc.encoder_downsampling_factor/target_fs,"\npixels predicted ahead:",self.model_cpc.encoder_downsampling_factor*self.hparams.steps_predicted,"\nseconds predicted ahead:",self.model_cpc.encoder_downsampling_factor*self.hparams.steps_predicted/target_fs,"\nRNN input size:",self.hparams.input_size//self.model_cpc.encoder_downsampling_factor)

    def forward(self, x):
        return self.model_cpc(x)
        
    def _step(self,data_batch, batch_idx, train):       
        if(self.hparams.finetune):
            preds = self.forward(data_batch[0])
            loss = self.criterion(preds,data_batch[1])
            self.log("train_loss" if train else "val_loss", loss)
            return {'loss':loss, "preds":preds.detach(), "targs": data_batch[1]}
        else:
            loss, acc = self.model_cpc.cpc_loss(data_batch[0],steps_predicted=self.hparams.steps_predicted,n_false_negatives=self.hparams.n_false_negatives, negatives_from_same_seq_only=self.hparams.negatives_from_same_seq_only, eval_acc=True)
            self.log("loss" if train else "val_loss", loss)
            self.log("acc" if train else "val_acc", acc)
            return loss
      
    def training_step(self, train_batch, batch_idx):
        if(self.hparams.linear_eval):
            _freeze_bn_stats(self)
        return self._step(train_batch,batch_idx,True)
        
    def validation_step(self, val_batch, batch_idx, dataloader_idx=0):
        return self._step(val_batch,batch_idx,False)
        
    def validation_epoch_end(self, outputs_all):
        if(self.hparams.finetune):
            for dataloader_idx,outputs in enumerate(outputs_all): #multiple val dataloaders
                preds_all = torch.cat([x['preds'] for x in outputs])
                targs_all = torch.cat([x['targs'] for x in outputs])
                if(self.hparams.finetune_dataset=="thew"):
                    preds_all = F.softmax(preds_all,dim=-1)
                    targs_all = torch.eye(len(self.lbl_itos))[targs_all].to(preds.device) 
                else:
                    preds_all = torch.sigmoid(preds_all)
                preds_all = preds_all.cpu().numpy()
                targs_all = targs_all.cpu().numpy()
                #instance level score
                res = eval_scores(targs_all,preds_all,classes=self.lbl_itos)
                
                idmap = self.val_dataset.get_id_mapping()
                preds_all_agg,targs_all_agg = aggregate_predictions(preds_all,targs_all,idmap,aggregate_fn=np.mean)
                res_agg = eval_scores(targs_all_agg,preds_all_agg,classes=self.lbl_itos)
                self.log_dict({"macro_auc_agg"+str(dataloader_idx):res_agg["label_AUC"]["macro"], "macro_auc_noagg"+str(dataloader_idx):res["label_AUC"]["macro"]})
                print("epoch",self.current_epoch,"macro_auc_agg"+str(dataloader_idx)+":",res_agg["label_AUC"]["macro"],"macro_auc_noagg"+str(dataloader_idx)+":",res["label_AUC"]["macro"])


    def on_fit_start(self):
        if(self.hparams.linear_eval):
            print("copying state dict before training for sanity check after training")   
            self.state_dict_pre = copy.deepcopy(self.state_dict().copy())

    
    def on_fit_end(self):
        if(self.hparams.linear_eval):
            sanity_check(self,self.state_dict_pre)
            
            
    def setup(self, stage):
        # configure dataset params
        chunkify_train = False
        chunk_length_train = self.hparams.input_size if chunkify_train else 0
        stride_train = self.hparams.input_size
        
        chunkify_valtest = True
        chunk_length_valtest = self.hparams.input_size if chunkify_valtest else 0
        stride_valtest = self.hparams.input_size//2

        train_datasets = []
        val_datasets = []
        test_datasets = []
        
        for i,target_folder in enumerate(self.hparams.data):
            target_folder = Path(target_folder)           
            
            df_mapped, lbl_itos,  mean, std = load_dataset(target_folder)
            # always use PTB-XL stats
            mean = np.array([-0.00184586, -0.00130277,  0.00017031, -0.00091313, -0.00148835,  -0.00174687, -0.00077071, -0.00207407,  0.00054329,  0.00155546,  -0.00114379, -0.00035649])
            std = np.array([0.16401004, 0.1647168 , 0.23374124, 0.33767231, 0.33362807,  0.30583013, 0.2731171 , 0.27554379, 0.17128962, 0.14030828,   0.14606956, 0.14656108])
            
            #specific for PTB-XL
            if(self.hparams.finetune and self.hparams.finetune_dataset.startswith("ptbxl")):
                if(self.hparams.finetune_dataset=="ptbxl_super"):
                    ptb_xl_label = "label_diag_superclass"
                elif(self.hparams.finetune_dataset=="ptbxl_all"):
                    ptb_xl_label = "label_all"
                    
                lbl_itos= np.array(lbl_itos[ptb_xl_label])
                
                def multihot_encode(x, num_classes):
                    res = np.zeros(num_classes,dtype=np.float32)
                    for y in x:
                        res[y]=1
                    return res
                    
                df_mapped["label"]= df_mapped[ptb_xl_label+"_filtered_numeric"].apply(lambda x: multihot_encode(x,len(lbl_itos)))
                    
            
            self.lbl_itos = lbl_itos
            tfms_ptb_xl_cpc = ToTensor() if self.hparams.normalize is False else transforms.Compose([Normalize(mean,std),ToTensor()])
            
            max_fold_id = df_mapped.strat_fold.max() #unfortunately 1-based for PTB-XL; sometimes 100 (Ribeiro)
            df_train = df_mapped[df_mapped.strat_fold<(max_fold_id-1 if self.hparams.finetune else max_fold_id)]
            df_val = df_mapped[df_mapped.strat_fold==(max_fold_id-1 if self.hparams.finetune else max_fold_id)]
            if(self.hparams.finetune):
                df_test = df_mapped[df_mapped.strat_fold==max_fold_id]
            train_datasets.append(TimeseriesDatasetCrops(df_train,self.hparams.input_size,num_classes=len(lbl_itos),data_folder=target_folder,chunk_length=chunk_length_train,min_chunk_length=self.hparams.input_size, stride=stride_train,transforms=tfms_ptb_xl_cpc,annotation=False,col_lbl ="label" if self.hparams.finetune else None,memmap_filename=target_folder/("memmap.npy")))
            val_datasets.append(TimeseriesDatasetCrops(df_val,self.hparams.input_size,num_classes=len(lbl_itos),data_folder=target_folder,chunk_length=chunk_length_valtest,min_chunk_length=self.hparams.input_size, stride=stride_valtest,transforms=tfms_ptb_xl_cpc,annotation=False,col_lbl ="label" if self.hparams.finetune else None,memmap_filename=target_folder/("memmap.npy")))
            if(self.hparams.finetune):
                test_datasets.append(TimeseriesDatasetCrops(df_test,self.hparams.input_size,num_classes=len(lbl_itos),data_folder=target_folder,chunk_length=chunk_length_valtest,min_chunk_length=self.hparams.input_size, stride=stride_valtest,transforms=tfms_ptb_xl_cpc,annotation=False,col_lbl ="label",memmap_filename=target_folder/("memmap.npy")))
            
            print("\n",target_folder)
            print("train dataset:",len(train_datasets[-1]),"samples")
            print("val dataset:",len(val_datasets[-1]),"samples")
            if(self.hparams.finetune):
                print("test dataset:",len(test_datasets[-1]),"samples")

        if(len(train_datasets)>1): #multiple data folders
            print("\nCombined:")
            self.train_dataset = ConcatDataset(train_datasets)
            self.val_dataset = ConcatDataset(val_datasets)
            print("train dataset:",len(self.train_dataset),"samples")
            print("val dataset:",len(self.val_dataset),"samples")
            if(self.hparams.finetune):
                self.test_dataset = ConcatDataset(test_datasets)
                print("test dataset:",len(self.test_dataset),"samples")
        else: #just a single data folder
            self.train_dataset = train_datasets[0]
            self.val_dataset = val_datasets[0]
            if(self.hparams.finetune):
                self.test_dataset = test_datasets[0]
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=4, shuffle=True, drop_last = True)
        
    def val_dataloader(self):
        if(self.hparams.finetune):#multiple val dataloaders
            return [DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=4),DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=4)]
        else:
            return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=4)

    def configure_optimizers(self):
        if(self.hparams.optimizer == "sgd"):
            opt = torch.optim.SGD
        elif(self.hparams.optimizer == "adam"):
            opt = torch.optim.AdamW
        else:
            raise NotImplementedError("Unknown Optimizer.")
        
        if(self.hparams.finetune and (self.hparams.linear_eval or self.hparams.train_head_only)):
            optimizer = opt(self.model_cpc.head.parameters(), self.lr, weight_decay=self.hparams.weight_decay)
        elif(self.hparams.finetune and self.hparams.discriminative_lr_factor != 1.):#discrimative lrs
            optimizer = opt([{"params":self.model_cpc.encoder.parameters(), "lr":self.lr*self.hparams.discriminative_lr_factor*self.hparams.discriminative_lr_factor},{"params":self.model_cpc.rnn.parameters(), "lr":self.lr*self.hparams.discriminative_lr_factor},{"params":self.model_cpc.head.parameters(), "lr":self.lr}],self.hparams.lr, weight_decay=self.hparams.weight_decay)
        else:
            optimizer = opt(self.parameters(), self.lr, weight_decay=self.hparams.weight_decay)

        return optimizer
        
    def load_weights_from_checkpoint(self, checkpoint):
        """ Function that loads the weights from a given checkpoint file. 
        based on https://github.com/PyTorchLightning/pytorch-lightning/issues/525
        """
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage,)
        pretrained_dict = checkpoint["state_dict"]
        model_dict = self.state_dict()
            
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

#####################################################################################################
#ARGPARSERS
#####################################################################################################
def add_model_specific_args(parser):
    parser.add_argument("--input-channels", type=int, default=12)
    parser.add_argument("--normalize", action='store_true', help='Normalize input using PTB-XL stats')
    parser.add_argument('--mlp', action='store_true', help="False: original CPC True: as in SimCLR")
    parser.add_argument('--bias', action='store_true', help="original CPC: no bias")
    parser.add_argument("--n-hidden", type=int, default=512)
    parser.add_argument("--gru", action="store_true")
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--steps-predicted", dest="steps_predicted", type=int, default=12)
    parser.add_argument("--n-false-negatives", dest="n_false_negatives", type=int, default=128)
    parser.add_argument("--skip-encoder", action="store_true", help="disable the convolutional encoder i.e. just RNN; for testing")
    parser.add_argument("--fc-encoder", action="store_true", help="use a fully connected encoder (as opposed to an encoder with strided convs)")
    parser.add_argument("--negatives-from-same-seq-only", action="store_true", help="only draw false negatives from same sequence (as opposed to drawing from everywhere)")
    parser.add_argument("--no-bn-encoder", action="store_true", help="switch off batch normalization in encoder")
    parser.add_argument("--dropout-head", type=float, default=0.5)
    parser.add_argument("--train-head-only", action="store_true", help="freeze everything except classification head (note: --linear-eval defaults to no hidden layer in classification head)")
    parser.add_argument("--lin-ftrs-head", type=str, default="[512]", help="hidden layers in the classification head")
    parser.add_argument('--no-bn-head', action='store_true', help="use no batch normalization in classification head")
    return parser

def add_default_args():
    parser = argparse.ArgumentParser(description='PyTorch Lightning CPC Training')
    parser.add_argument('--data', metavar='DIR',type=str,
                        help='path(s) to dataset',action='append')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 0.)',
                        dest='weight_decay')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

    parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained checkpoint (default: none)')
    parser.add_argument('--optimizer', default='adam', help='sgd/adam')#was sgd
    parser.add_argument('--output-path', default='.', type=str,dest="output_path",
                        help='output path')
    parser.add_argument('--metadata', default='', type=str,
                        help='metadata for output')
    
    parser.add_argument("--gpus", type=int, default=1, help="number of gpus")
    parser.add_argument("--num-nodes", dest="num_nodes", type=int, default=1, help="number of compute nodes")
    parser.add_argument("--precision", type=int, default=16, help="16/32")
    parser.add_argument("--distributed-backend", dest="distributed_backend", type=str, default=None, help="None/ddp")
    parser.add_argument("--accumulate", type=int, default=1, help="accumulate grad batches (total-bs=accumulate-batches*bs)")
        
    parser.add_argument("--input-size", dest="input_size", type=int, default=16000)
    
    parser.add_argument("--finetune", action="store_true", help="finetuning (downstream classification task)",  default=False )
    parser.add_argument("--linear-eval", action="store_true", help="linear evaluation instead of full finetuning",  default=False )
    
    parser.add_argument(
        "--finetune-dataset",
        type=str,
        help="thew/ptbxl_super/ptbxl_all",
        default="thew"
    )
    
    parser.add_argument(
        "--discriminative-lr-factor",
        type=float,
        help="factor by which the lr decreases per layer group during finetuning",
        default=0.1
    )
    
    
    parser.add_argument(
        "--lr-find",
        action="store_true",
        help="run lr finder before training run",
        default=False
    )
    
    return parser
             
###################################################################################################
#MAIN
###################################################################################################
if __name__ == '__main__':
    parser = add_default_args()
    parser = add_model_specific_args(parser)
    hparams = parser.parse_args()
    hparams.executable = "cpc"

    if not os.path.exists(hparams.output_path):
        os.makedirs(hparams.output_path)
        
    model = LightningCPC(hparams)
    
    if(hparams.pretrained!=""):
        print("Loading pretrained weights from",hparams.pretrained)
        model.load_weights_from_checkpoint(hparams.pretrained)


    logger = TensorBoardLogger(
        save_dir=hparams.output_path,
        #version="",#hparams.metadata.split(":")[0],
        name="")
    print("Output directory:",logger.log_dir)    
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(logger.log_dir,"best_model"),#hparams.output_path
        save_top_k=1,
		save_last=True,
        verbose=True,
        monitor='macro_auc_agg0' if hparams.finetune else 'val_loss',#val_loss/dataloader_idx_0
        mode='max' if hparams.finetune else 'min',
        prefix='')
    lr_monitor = LearningRateMonitor()

    trainer = pl.Trainer(
        #overfit_batches=0.01,
        auto_lr_find = hparams.lr_find,
        accumulate_grad_batches=hparams.accumulate,
        max_epochs=hparams.epochs,
        min_epochs=hparams.epochs,
        
        default_root_dir=hparams.output_path,
        
        num_sanity_val_steps=0,
        
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        callbacks = [],#lr_monitor],
        benchmark=True,
    
        gpus=hparams.gpus,
        num_nodes=hparams.num_nodes,
        precision=hparams.precision,
        distributed_backend=hparams.distributed_backend,
        
        progress_bar_refresh_rate=0,
        weights_summary='top',
        resume_from_checkpoint= None if hparams.resume=="" else hparams.resume)
        
    if(hparams.lr_find):#lr find
        trainer.tune(model)
        
    trainer.fit(model)
    

