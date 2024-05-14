import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from cnrs_project.ENG.TimeLLM import Model

import time
import random
import numpy as np
import os
import pandas as pd
from cnrs_project.ENG.linker_ENG import linker
from cnrs_project.ENG.functional import Logger
from cnrs_project.ENG.data_factory import MMDataset, out_of_type_train_test
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from cnrs_project.ENG.functional import FocalLoss, vali
import sys


os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import del_files, EarlyStopping, adjust_learning_rate

parser = argparse.ArgumentParser(description='Time-LLM')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--task_name', type=str, required=False, default='classification',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=False, default='none', help='prefix when saving test results')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=4, help='patch length')
parser.add_argument('--stride', type=int, default=2, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=128, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)

parser.add_argument('--test_crisis', type=str, default='Hurricane',
                    choices=['Hurricane', 'WildFire', 'Flood', 'Earthquake'])


args = parser.parse_args()
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='../ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)
directory_nlp = '/home/eee/qzz/datasets/CNRS/Crisis-TS-NLP/English_Corpus'
directory_time_series = '/home/eee/qzz/datasets/CNRS/Crisis-TS-NLP/MeteoData-US'
path_knowledge = '/home/eee/qzz/datasets/CNRS/Crisis-TS-NLP/crisis_knowledge_EN.csv'
knowledge = pd.read_csv(path_knowledge, sep='\t')  # (9, 3), 3 columns represent Crisis, Places, Path_name (tweets)

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_sl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.seq_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des, ii)

    train_df, test_df = out_of_type_train_test(knowledge, args.test_crisis,
                                               linker, directory_nlp, directory_time_series)

    # Map label to int
    dic_cat_labels = {'Not_Crisis_period': 0, 'Predictible_crisis': 1, 'Sudden_crisis': 2}
    train_df['label'] = train_df['label'].map(dic_cat_labels)
    test_df['label'] = test_df['label'].map(dic_cat_labels)

    # Train val split
    train_df, vali_df = train_test_split(train_df, test_size=0.2, random_state=42, shuffle=True)
    train_data, vali_data, test_data = MMDataset(train_df), MMDataset(vali_df), MMDataset(test_df)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    vali_loader = DataLoader(vali_data, batch_size=args.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False)

    # ZZ: Use the cumstom TimeLLM for classification
    model = Model(args, n_classes=len(dic_cat_labels), n_vars=11).float()

    path = os.path.join(args.checkpoints,
                        setting + '-' + args.model_comment)  # unique checkpoint saving path


    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    log_dir = path + '/' + 'log.txt'
    sys.stdout = Logger(log_dir)

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)

    # ToDo: Classification loss
    criterion = FocalLoss(gamma=5., alpha=0.25, num_classes=len(dic_cat_labels))  # torch.nn.CrossEntropyLoss()

    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim, scheduler)

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, (batch_date, batch_text, batch_win, batch_lab) in tqdm(enumerate(train_loader)):
            iter_count += 1
            model_optim.zero_grad()

            # batch_text = batch_text.to(accelerator.device)
            batch_win = batch_win.float().to(accelerator.device)
            batch_lab = batch_lab.to(torch.int64).to(accelerator.device)

            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_text, batch_win)[0]
                    else:
                        outputs = model(batch_text, batch_win)
                    loss = criterion(outputs, batch_lab)
                    train_loss.append(loss.item())
            else:
                if args.output_attention:
                    outputs = model(batch_text, batch_win)[0]
                else:
                    outputs = model(batch_text, batch_win)
                loss = criterion(outputs, batch_lab)
                train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                accelerator.print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                accelerator.backward(loss)
                model_optim.step()

            if args.lradj == 'TST':
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)

        # Revise to align with classification
        vali_loss, vali_cr = vali(args, accelerator, model, vali_loader, criterion, dic_cat_labels)
        test_loss, test_cr = vali(args, accelerator, model, test_loader, criterion, dic_cat_labels)

        accelerator.print(
            "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f}".format(
                epoch + 1, train_loss, vali_loss, test_loss))

        accelerator.print(' ----- Vali classification report -----\n', vali_cr)
        accelerator.print(' ----- Test classification report -----\n', test_cr)

        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break

        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

        else:
            accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

accelerator.wait_for_everyone()
# if accelerator.is_local_main_process:
#     path = './checkpoints'  # unique checkpoint saving path
#     del_files(path)  # delete checkpoint files
#     accelerator.print('success delete checkpoints')

