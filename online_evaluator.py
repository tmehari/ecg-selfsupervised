import math
import pdb
import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.functional import accuracy
from torch.nn import functional as F
from clinical_ts.eval_utils_cafa import eval_scores, eval_scores_bootstrap
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from torch.nn.modules.linear import Linear
from copy import deepcopy
from clinical_ts.create_logger import create_logger
from tqdm import tqdm

logger = create_logger(__name__)


class SSLOnlineEvaluator(pl.Callback):  # pragma: no-cover

    def __init__(self, drop_p: float = 0.0, hidden_dim: int = 1024, z_dim: int = None, num_classes: int = None, lin_eval_epochs=5, eval_every=10, mode="linear_evaluation", discriminative=True, verbose=False):
        """
        Attaches a MLP for finetuning using the standard self-supervised protocol.
        Example::
            from pl_bolts.callbacks.self_supervised import SSLOnlineEvaluator
            # your model must have 2 attributes
            model = Model()
            model.z_dim = ... # the representation dim
            model.num_classes = ... # the num of classes in the model
        Args:
            drop_p: (0.2) dropout probability
            hidden_dim: (1024) the hidden dimension for the finetune MLP
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.drop_p = drop_p
        self.optimizer = None
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.macro = 0
        self.best_macro = 0
        self.lin_eval_epochs = lin_eval_epochs
        self.eval_every = eval_every
        self.discriminative = discriminative
        self.verbose = verbose
        if mode == "linear_evaluation":
            self.mode = mode
        elif mode == "fine_tuning":
            self.mode = mode
        else:
            raise("mode " + str(mode) + " unknown")

    def get_representations(self, features, x):
        """
        Override this to customize for the particular model
        Args:
            pl_module:
            x:
        """
        if len(x) == 2 and isinstance(x, list):
            x = x[0]

        representations = features(x)

        if (isinstance(representations, list) or isinstance(representations, tuple)):
            representations = representations[0]

        representations = representations.reshape(representations.size(0), -1)
        return representations

    def to_device(self, batch, device):
        x, y = batch
        return x, y

    def put_on_device(self, batch, device, new_type):
        x, y = batch
        x = x.type(new_type).to(device)
        y = y.type(new_type).to(device)
        return x, y

    def on_sanity_check_start(self, trainer, pl_module):
        self.val_ds_size = len(trainer.val_dataloaders[0].dataset)
        self.last_batch_id = len(trainer.val_dataloaders[0])-1

    def on_sanity_check_end(self, trainer, pl_module):
        self.macro = 0

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    #def on_validation_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # reset mlp after each epoch to get fresh linear evaluation values at every epoch
        if pl_module.epoch % self.eval_every == 0 and batch_idx == 0 and dataloader_idx == 0:
            new_type, device, valid_loader, features, linear_head, optimizer = self.online_train_setup(
                pl_module, trainer)

            loss_per_epoch = []
            macro_per_epoch = []
            linear_head2 = deepcopy(linear_head)
            for epoch in tqdm(range(self.lin_eval_epochs)):

                total_loss_one_epoch, linear_head = self.train_one_epoch(
                    valid_loader, features, linear_head, optimizer, device, new_type)
                
                if self.verbose:
                    loss_per_epoch.append(total_loss_one_epoch)
                    macro, total_loss = self.eval_model(
                        trainer, features, linear_head, device, new_type)
                    macro_per_epoch.append(macro)
                    logger.info("macro at epoch "+str(epoch) + ": " + str(macro))
                    logger.info("train loss at epoch "+str(epoch) + ": " + str(total_loss_one_epoch))
                    logger.info("test loss at epoch "+str(epoch) + ": " + str(total_loss))

            macro, total_loss = self.eval_model(trainer, features, linear_head, device, new_type)
            self.log_values(trainer, pl_module, macro, total_loss)
           
    def online_train_setup(self, pl_module, trainer):
        new_type = pl_module.type()
        device = pl_module.get_device()
        valid_loader = trainer.val_dataloaders[1]
        if self.mode == "linear_evaluation":
            lr = 8e-3 *(valid_loader.batch_size/256)
        else: 
            lr = 8e-5 *(valid_loader.batch_size/256)
        # print("using lr:", lr)
        # print("using batch size: ", valid_loader.batch_size)
        wd = 1e-1
        features = deepcopy(pl_module.get_model())
        linear_head = Linear(
            features.l1.in_features, self.num_classes, bias=True).type(new_type)
        if self.mode == "linear_evaluation":
            optimizer = torch.optim.AdamW(
                linear_head.parameters(), lr=lr, weight_decay=wd)
        else:
            if not self.discriminative:
                optimizer = torch.optim.AdamW([
                    {"params": features.parameters()}, {"params": linear_head.parameters()}], lr=lr, weight_decay=wd)
            else:
                lr = (8e-3*(valid_loader.batch_size/256))
                param_dict = dict(features.named_parameters())
                keys = param_dict.keys()
                weight_layer_nrs = set()
                for key in keys:
                    if "features" in key:
                        # parameter names have the form features.x
                        weight_layer_nrs.add(key[9])
                weight_layer_nrs = sorted(weight_layer_nrs, reverse=True)
                features_groups = []
                while len(weight_layer_nrs) > 0:
                    if len(weight_layer_nrs) > 1:
                        features_groups.append(list(filter(
                            lambda x: "features." + weight_layer_nrs[0] in x or "features." + weight_layer_nrs[1] in x,  keys)))
                        del weight_layer_nrs[:2]
                    else:
                        features_groups.append(
                            list(filter(lambda x: "features." + weight_layer_nrs[0] in x,  keys)))
                        del weight_layer_nrs[0]
                # linears = list(filter(lambda x: "l" in x, keys)) # filter linear layers
                # groups = [linears] + features_groups
                optimizer_param_list = []
                tmp_lr = lr
                optimizer_param_list.append(
                        {"params": linear_head.parameters(), "lr": tmp_lr})
                tmp_lr /= 4
                for layers in features_groups:
                    layer_params = [param_dict[param_name]
                                    for param_name in layers]
                    optimizer_param_list.append(
                        {"params": layer_params, "lr": tmp_lr})
                    tmp_lr /= 4
                optimizer = torch.optim.AdamW(optimizer_param_list, lr=lr, weight_decay=wd)
            
        return new_type, device, valid_loader, features, linear_head, optimizer

    def train_one_epoch(self, valid_loader, features, linear_head, optimizer, device, new_type):
        linear_head.train()
        if self.mode == "linear_evaluation":
            # we dont want to update things like batchnorm statistics in linear evaluation
            features.eval()
        else:
            features.train()
        total_loss_one_epoch = 0
        for cur_batch in valid_loader:
            x, y = self.put_on_device(
                cur_batch, device, new_type)
            if self.mode == "linear_evaluation":
                with torch.no_grad():
                    representations = self.get_representations(
                        features, x)
            else:
                with torch.enable_grad():
                    representations = self.get_representations(
                        features, x)
            # forward pass
            with torch.enable_grad():
                mlp_preds = linear_head(representations)
                mlp_loss = F.binary_cross_entropy_with_logits(
                    mlp_preds, y)
                # update finetune weights
                optimizer.zero_grad()
                mlp_loss.backward()
                optimizer.step()
            total_loss_one_epoch += mlp_loss.item()
        return total_loss_one_epoch, linear_head

    def eval_model(self, trainer, features, linear_head, device, new_type):
        features.eval()
        preds = []
        labels = []
        total_loss = 0
        test_loader = trainer.val_dataloaders[2]
        for cur_batch in test_loader:
            x, y = self.put_on_device(
                cur_batch, device, new_type)
            with torch.no_grad():
                representations = self.get_representations(features, x)
                mlp_preds = torch.sigmoid(
                    linear_head(representations))
                preds.append(mlp_preds.cpu())
                labels.append(y.cpu())
                total_loss += F.binary_cross_entropy_with_logits(
                    mlp_preds, y)
        preds = torch.cat(preds).numpy()
        labels = torch.cat(labels).numpy()
        macro = roc_auc_score(labels, preds)
        return macro, total_loss

    def log_values(self, trainer, pl_module, macro, total_loss):
        self.best_macro = macro if macro > self.best_macro else self.best_macro
        if self.mode == "linear_evaluation":
            log_key = "le"
        else:
            log_key = "ft"
        metrics = {log_key + '_mlp/loss': total_loss,
                   log_key + '_mlp/macro': macro, log_key + '_mlp/best_macro': self.best_macro}
        pl_module.logger.log_metrics(metrics, step=trainer.global_step)

    def __str__(self):
        return self.mode+"_callback"