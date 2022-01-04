# Self-supervised representation learning from 12-lead ECG data
This is  the offical code repository accompanying our paper on **Self-supervised representation learning from 12-lead ECG data**.

For a detailed description of technical details and experimental results, please refer to our paper:

Temesgen Mehari, and Nils Strodthoff, [Self-supervised representation learning from 12-lead ECG data](https://doi.org/10.1016/j.compbiomed.2021.105114), Computers in Biology and Medicine **141** (2022) 105114.
    
    @article{Mehari:2021Self,
        doi = {10.1016/j.compbiomed.2021.105114},
        url = {https://doi.org/10.1016/j.compbiomed.2021.105114},
        year = {2022},
        month = feb,
        publisher = {Elsevier {BV}},
        volume = {141},
        pages = {105114},
        author = {Temesgen Mehari and Nils Strodthoff},
        title = {Self-supervised representation learning from 12-lead {ECG} data},
        journal = {Computers in Biology and Medicine},
        eprint = {2103.12676},
        archivePrefix={arXiv},
        primaryClass={eess.SP}
    }

## Usage information
### Preparation
1. install dependencies from `ecg_selfsupervised.yml` by running `conda env create -f ecg_selfsupervised.yml` and activate the environment via `conda activate ecg_selfsupervised`
2. follow the instructions in `data_preprocessing.ipynb` on how to download and preprocess the ECG datasets; in the following, we assume for definiteness that the preprocessed dataset folders can be found at `./data/cinc`,`./data/zheng`,`./data/ribeiro` and `./data/ptb_xl`.

### A1. Pretraining CPC on All
Pretrain a CPC model on the full dataset collection (will take about 6 days on a Tesla V100 GPU):
`python main_cpc_lightning.py --data ./data/cinc --data ./data/zheng --data ./data/ribeiro --normalize --epochs 1000 --output-path=./runs/cpc/all --lr 0.0001 --batch-size 32 --input-size 1000 --fc-encoder --negatives-from-same-seq-only`

### A2. Finetuning CPC on PTB-XL
1. Finetune just the classification head:
`python main_cpc_lightning.py --data ./data/ptb_xl --normalize --epochs 50 --output-path=./runs/cpc/all_ptbxl --lr 0.001 --batch-size 128 --input-size 250 --finetune --pretrained ./runs/cpc/all/version_0/best_model.ckpt --finetune-dataset ptbxl_all --fc-encoder --train-head-only`
2. Finetune the entire model with discriminative learning rates:
`python main_cpc_lightning.py --data ./data/ptb_xl --epochs 20 --output-path=./runs/cpc/all_ptbxl --lr 0.0001 --batch-size 128 --normalize --input-size 250 --finetune --finetune-dataset ptbxl_all --fc-encoder --pretrained ./runs/cpc/all_ptbxl/version_0/best_model.ckpt`

### B1. Pretraing SimCLR, BYOL, and SwAV  on All
Pretrain a xresnet1d50 model on the full dataset by running the corresponding lightning module:

SimCLR: `python custom_simclr_bolts.py --batch_size 4096 --epochs 2000 --precision 16 --trafos RandomResizedCrop TimeOut
--datasets ./data/cinc ./data/zheng ./data/ribeiro --log_dir=experiment_logs`

BYOL: `python custom_byol_bolts.py --batch_size 4096 --epochs 2000 --precision 16 --trafos RandomResizedCrop TimeOut
--datasets ./data/cinc ./data/zheng ./data/ribeiro --log_dir=experiment_logs`

SwAV: `python custom_swav_bolts.py --batch_size 4096 --epochs 2000 --precision 16 --trafos RandomResizedCrop TimeOut
--datasets ./data/cinc ./data/zheng ./data/ribeiro --log_dir=experiment_logs`

Transformations can be combined by sequentially attaching as shown in the command above. One can choose from [GaussianNoise, GaussianBlur, RandomResizedCrop, TimeOut, ChannelResize, DynamicTimeWarp, BaselineWander, PowerlineNoise, EMNoise, BaselineShift]. The Transformations are described in detail in the appendix of the paper. 
All of the lightning modules come with some (hyper)parameters. Feel free to run `python *lightning_module* --help` to get the descriptions of the parameters. 

The results of the runs will be written to the log_dir in the following form:

experiment\_logs </br>
      → Wed Dec  2 17:06:54 2020\_simclr\_696\_RRC TO </br>
            → checkpoints </br>
                  → epoch=1654.ckpt </br>
                  → model.ckpt </br>
            → events.out.tfevents.1606925231 </br>
            → hparams.yaml </br>
      

2 directories, 5 files

While hparams.yaml and the event file store the hyperparameters and tensorboard logs of the run, respectively, we store the trained models in the checkpoints directory. We store two models, the best model according to the validation loss and the last model after training.

### B2. Finetuning SimCLR, BYOL and SwAV (and CPC) on PTB-XL
Finetune a given model by running 

`python eval.py --method simclr --model_file "experiment_logs/Wed Dec  2 17:06:54 2020\_simclr\_696\_RRC TO/checkpoints/model.ckpt" --batch_size 128 --use_pretrained --f_epochs 50 --dataset ./data/ptb_xl` 

and similarly perform a linear evaluation by running 

`python eval.py --method simclr --model_file "experiment_logs/Wed Dec  2 17:06:54 2020\_simclr\_696\_RRC TO/checkpoints/model.ckpt" --batch_size 128 --use_pretrained --l_epochs 50 --linear_evaluation --dataset ./data/ptb_xl`

The most important parameters are the location of the model you want to finetune and the method that was used during pretraining. Again, consider running python eval.py --help to get yourself an overview of the possible parameters. The results are written to the model directory:

experiment\_logs </br>
      → Wed Dec  2 17:06:54 2020\_simclr\_696\_RRC TO </br>
            → checkpoints </br>
                  → epoch=1654.ckpt </br>
                  → model.ckpt </br>
                  → n=1\_f=8\_fin_finetuned </br>
                        → finetuned.pt </br>
                  → n=1\_f=8\_res\_fin.pkl </br>
            → events.out.tfevents.1606925231 </br>
            → hparams.yaml 
      

3 directories, 7 files

The script prints the results and saves the finetuned model in a new folder next to the pretrained model and a pickle file containing the results of the evaluation.




## Pretrained models
For each method (SimCLR, BYOL, CPC), we provide the best-performing pretrained model after pretraining on *All*: [link to datacloud](https://cloud.uol.de/s/WyfdBXt64DWJaSc). More models are available from the authors upon request.

