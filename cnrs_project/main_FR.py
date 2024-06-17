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

fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--model_comment', type=str, required=False, default='none', help='prefix when saving test results')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--checkpoints', type=str, default='./checkpoints/FR', help='location of model checkpoints')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=3, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')  # 0.1
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=4, help='patch length')
parser.add_argument('--stride', type=int, default=2, help='stride')
parser.add_argument('--llm_model', type=str, default='Roberta', help='LLM model') # LLAMA, GPT2, BERT, Roberta
parser.add_argument('--llm_dim', type=int, default='768', help='LLM model dimension')# LLama7b:4096; # GPT2-small:768; BERT-base/Roberta:768
parser.add_argument('--llm_layers', type=int, default=6)  # 6 / 12

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=32, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=1e-2, help='optimizer learning rate')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')  # type1
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

parser.add_argument('--test_crisis', type=str, default='Flood',
                    choices=['Fire', 'Flood', 'Storms', 'Hurricane', 'Explosion', 'Collapse', 'ATTACK'])


def remove_values_from_list(the_list, val):
    return [value for value in the_list if value != val]


args = parser.parse_args()
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='../ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)


# directory_nlp = '/home/eee/qzz/datasets/CNRS/Crisis-TS-NLP/English_Corpus'
# directory_time_series = '/home/eee/qzz/datasets/CNRS/Crisis-TS-NLP/MeteoData-FR'
# path_knowledge = '/home/eee/qzz/datasets/CNRS/Crisis-TS-NLP/crisis_knowledge_FR.csv'
# knowledge = pd.read_csv(path_knowledge, sep=',')  # (10, 3), 3 columns represent Crisis, Places, Path_name (tweets)

Test_crisis = [args.test_crisis]
file_data = '/home/eee/qzz/datasets/CNRS/Crisis-TS-NLP/multi_modal_french.csv'

mm_data = pd.read_csv(file_data, sep="\t")
#the window is saved as string we need to rechange the type to np array
list_of_time_series_data = []


for row in mm_data["Window"]:
	a = row.replace("[","")
	a = a.replace("]","")
	a = a.replace("\n"," ")
	row = []
	total = []
	while '  ' in a :
		a = a.replace('  ',' ')
	a = a.split(" ")
	a = remove_values_from_list(a,'')
	#size of the window
	for j in range(24) :
		#number of parameters
		row=a[6*j:6*j+6]
		total.append(row)
	list_of_time_series_data.append(np.array(total))

mm_data['Window'] = list_of_time_series_data


for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_ll{}_lr{}_la{}_do{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_se{}_{}'.format(
        args.test_crisis,
        args.llm_model,
        args.llm_layers,
        args.learning_rate,
        args.lradj,
        args.dropout,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        fix_seed,
        ii)

    train_df, test_df = mm_data[~mm_data.Crisis_Type.isin(Test_crisis)], mm_data[mm_data.Crisis_Type.isin(Test_crisis)]

    # Map label to int
    dic_cat_labels = {0: 'Not_Crisis_period', 1: 'Ecological_crisis', 2: 'Sudden_Crisis'}
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

    # Revise to align with classification
    test_loss, test_cr = vali(args, accelerator, model, test_loader, criterion, dic_cat_labels)
    accelerator.print("Before Training: Test Loss: {}".format(test_loss))
    accelerator.print(' ----- Test classification report -----\n', test_cr)

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

