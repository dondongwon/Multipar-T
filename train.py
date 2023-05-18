import os 
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from options import parser
from trainer import Trainer, TrainerEnsemble
import utils
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import pdb
from joblib import Parallel, delayed
import random
import utils 
import pickle 
import time
from losses import SupConLoss, FocalLoss
import torchvision
from imbalanced import ImbalancedDatasetSampler
import sklearn 

    
args = parser.parse_args()

from model import *

########################################################################
#Group Num Specific Datasets & Dataloader
########################################################################

context_secs = int(args.context_secs) 
if args.data == 'roomreader':
    group_nums_dict = utils.rr_group_nums
    import dataset_vid
    from dataset_vid import RoomReader as DS
    import sys 
    sys.modules['dataset'] = dataset_vid
    input_feats = len(utils.roomreader_desired_feats)
    context_frames = (context_secs) * int(args.get_n_frames_per_sec)


group_ids = group_nums_dict[int(args.group_num)]

video_feat = args.video_feat

if args.pickled_dataset: 
    start_time = time.time()
    print('start loading data')

    pickeled_dataset_dst = "{PATH TO DIR}"

    if args.data == 'roomreader': 
        with open(pickeled_dataset_dst, 'rb') as handle: 
            group_all_dataset = pickle.load(handle)    
    if args.data == 'roomreaderNoVid': 
        with open(pickeled_dataset_dst, 'rb') as handle:
            group_all_dataset = pickle.load(handle)
    print("--- %s seconds ---" % (time.time() - start_time))
    print('done instantiating dataset... concatenating')

else:
    if args.data_split == 'debug':
        group_ids = [group_nums_dict[int(args.group_num)][0]]
        val_group_id = [group_nums_dict[int(args.group_num)][0]]
        test_group_id =  [group_nums_dict[int(args.group_num)][0]]

    group_all_dataset = []
    for group_id in tqdm(group_ids):   
        group_specific_dataset = DS(group_id = group_id, context_secs = context_secs, get_n_frames_per_sec = args.get_n_frames_per_sec, video_feat = video_feat) 
        group_all_dataset.append(group_specific_dataset)
    
if args.data_split == 'debug':
    train_set = group_all_dataset[0]
    val_set = group_all_dataset[0]
    val_group_id = val_set.group_id
    test_set = group_all_dataset[0]

    test_group_id = test_set.group_id

if args.data_split == 'bygroup_multitest':

    random.Random(int(args.seed)).shuffle(group_all_dataset)
    train_set = torch.utils.data.ConcatDataset(group_all_dataset[-6:])
    val_set =  torch.utils.data.ConcatDataset(group_all_dataset[-6:-3])
    # val_group_id = val_set.group_id

    test_sets = group_all_dataset[-3:]
    test_group_id = str([t_set.group_id for t_set in test_sets])
    test_set = torch.utils.data.ConcatDataset(test_sets)

if args.data_split == 'bygroup':

    random.Random(int(args.seed)).shuffle(group_all_dataset)
    train_set = torch.utils.data.ConcatDataset(group_all_dataset[-4:])
    val_set =  torch.utils.data.ConcatDataset(group_all_dataset[-4:-1])
    test_set = group_all_dataset[-1]
    test_group_id = test_set.group_id
    
    
    print('done setting up splits')

if args.data_split == 'bysample':
    all_data = torch.utils.data.ConcatDataset(group_all_dataset)
    test_len = len(all_data)//10
    val_len = len(all_data)//10
    train_len = len(all_data) - (test_len + val_len)
    train_set, val_set, test_set = torch.utils.data.random_split(all_data, (train_len, val_len, test_len), generator=torch.Generator().manual_seed(int(args.seed))) 

batch_size = int(args.batch_size)
epochs = int(args.epochs)


if args.oversampling:
    sampler = ImbalancedDatasetSampler(train_set)
    trainloader = DataLoader(train_set, batch_size = batch_size, sampler = sampler, drop_last=False, num_workers = 0)


else:
    trainloader = DataLoader(train_set, batch_size = batch_size, shuffle = True, drop_last=False, num_workers = 0)
#for idx, batch in enumerate(tqdm(trainloader)): pass

 
valloader = DataLoader(val_set, batch_size = batch_size, shuffle = False, drop_last=False,num_workers = 0)
testloader = DataLoader(test_set, batch_size = batch_size, shuffle = False, drop_last=False,num_workers = 0)
print('Done loading data!')


########################################################################
#logging
########################################################################
group_num = int(args.group_num)
model_name = args.model_name
print("Model Chosen: {}".format(model_name))

model_unique = "{model}".format(model = model_name + "_seed{}_lr{}_test{}".format(int(args.seed), float(args.lr), test_group_id))

weight_dir = "./model_weights/{}/{}/group_{}".format(args.save_dir, args.train_level, group_num)
if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)

log_dir = "./logs/{}/{}/group_{}".format(args.save_dir, args.train_level, group_num)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


log_path = log_dir + "/{model}.log".format(model = model_unique)
json_path = log_dir + "/{model}.json".format(model = model_unique)
weight_path = weight_dir + "/{model}.pth".format(model = model_unique)
print(log_path)

########################################################################
#loss
########################################################################

print('loss')
device = torch.device("cuda:1") # if torch.cuda.is_available() else "cpu")


if args.loss == "classify":
    
    criterion = nn.CrossEntropyLoss()
    label_levels = 4


if args.loss == "weighted_classify":

    y = []
    for i in range(len(train_set)):
        print(i)
        sample = train_set[i]
        try:
            y.append(sample['eng'][:,-1])
        except Exception:
            pdb.set_trace()
    
    y = np.stack(y)
    y = np.floor(np.clip(np.stack(y) +2 , a_min = 0, a_max = 3)).flatten()
    y = np.append(y, [0,1,2,3])
    class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight = "balanced", classes = np.unique(y), y = y)
    class_weights = torch.tensor(class_weights,dtype=torch.float).to(device)
 
    
    criterion = nn.CrossEntropyLoss(weight = class_weights)
    label_levels = 4

if args.loss == "focal":
    criterion = FocalLoss(gamma = 10)
    label_levels = 4

    print('loss chosen classification')

if args.loss == "mse":
    criterion = nn.MSELoss() 
    label_levels = None 

if args.loss == "ordinal":
    label_levels = 3
    criterion = nn.MSELoss() 


########################################################################
#model selection & parallelization
########################################################################


print("model selection & parallelization")


if args.train_level == 'group':
    if args.model_name == 'MultipartyTransformer':
        model = MultipartyTransformer(behavior_dims = int(args.behavior_dims), input_feats=2183, out_feats = 1, label_levels = label_levels)

    if args.model_name == 'Multiparty_GAT':
        model = Multiparty_GAT(nfeat=1564, nhid=64, nclass=(group_num - 1), dropout=0.5, alpha=0.2, nheads=3)


if args.train_level == 'individual':
    if args.model_name == 'TEMMA':
        model = TEMMA(input_feats=2048, out_feats = 1, label_levels = label_levels, context_frames = context_frames, nhead = 4)

    if args.model_name == 'ConvLSTM':
        model = ConvLSTM(input_feats=2048, out_feats = 1, label_levels = label_levels)
        print('selected {}'.format(args.model_name))

    if args.model_name == 'OCTCNNLSTM':
        model = OCTCNNLSTM(input_feats=input_feats, out_feats = 1, label_levels = label_levels)

    if args.model_name == 'BOOT':
        model = BOOT(input_feats=input_feats, out_feats = 1, label_levels = label_levels)

    if args.model_name == 'EnsModel':
        model = EnsModel()

    if args.model_name == 'HTMIL':
        model = HTMIL()

    
if args.parallel: # and args.data_split != 'debug'
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids = list(range(torch.cuda.device_count())))


model.to(device)

print('Done loading model!')


########################################################################
#optimizer
########################################################################

if 'Ens' in args.model_name: 
    optimizer1 = torch.optim.AdamW(model.m1.parameters(), lr=float(args.lr))
    optimizer2 = torch.optim.AdamW(model.m2.parameters(), lr=float(args.lr))
    optimizer3 = torch.optim.AdamW(model.m3.parameters(), lr=float(args.lr))
    optimizer4 = torch.optim.AdamW(model.m4.parameters(), lr=float(args.lr))

    optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]

    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=5, gamma=0.1)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=5, gamma=0.1)
    scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=5, gamma=0.1)
    scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=5, gamma=0.1)

    schedulers = [scheduler1, scheduler2, scheduler3, scheduler4]

    # optimizer = torch.optim.AdamW(params, lr=float(args.lr))

else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


########################################################################
#train
########################################################################

if 'Ens' in args.model_name: 
    trainer = TrainerEnsemble(
        model=model,
        criterion=criterion,
        optimizers=optimizers,
        schedulers=schedulers,
        log_path = log_path, 
        weight_path = weight_path,
        json_path = json_path,
        args = args
        )

else:
    trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    log_path = log_path, 
    weight_path = weight_path,
    json_path = json_path,
    args = args
    )

losses = trainer.fit(trainloader, valloader, testloader, epochs)