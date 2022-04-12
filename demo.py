import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
from utils.tools import dotdict


fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

args = dotdict()

# basic config
args.is_training = 1
args.model_id = 'archxixia_ms_48_48'
args.model = 'Autoformer'

# data loader
args.data = 'ETTh2ms1f2'
args.root_path = './data/ETT/'
args.data_path = 'archxixia_ms.csv'
args.feature = 'ms'
args.target = 'OT'
args.freq = 'ms'
args.checkpoints = './checkpoints/'
# TODO NOTE important adding:
args.train_ratio = 0.8
args.dev_ratio = 0.1
args.test_ratio = 0.1

# forecasting task
args.seq_len = 48
args.label_len = 48
args.pred_len = 48

# model define
args.bucket_size = 4
args.n_hashes = 4
args.enc_in = 1
args.dec_in = 1
args.c_out = 1
args.d_model = 512
args.n_heads = 8
args.e_layers = 2
args.d_layers = 1
args.d_ff = 2048
args.moving_avg = 25              # window size of moving average
args.factor = 1               # attn factor
args.distill = True
args.dropout = 0.05
args.embed = 'timeF'
args.activation = 'gelu'
args.out_attention = False
args.do_predict = False

# optimization
args.num_workers = 0
args.itr = 2
args.train_epochs = 6
args.batch_size = 4
args.patience = 3
args.learning_rate = 0.0001
args.des = 'exp'
args.loss = 'mse'
args.lradj = 'type1'
args.use_amp = False

# GPU
args.use_gpu = True if torch.cuda.is_available() else False
args.gpu = 0
args.use_multi_gpu = False
args.devices = '0,1,2,3'

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                              args.model,
                                                                                              args.data,
                                                                                              args.features,
                                                                                              args.seq_len,
                                                                                              args.label_len,
                                                                                              args.pred_len,
                                                                                              args.d_model,
                                                                                              args.n_heads,
                                                                                              args.e_layers,
                                                                                              args.d_layers,
                                                                                              args.d_ff,
                                                                                              args.factor,
                                                                                              args.embed,
                                                                                              args.distil,
                                                                                              args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
