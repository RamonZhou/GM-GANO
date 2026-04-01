import torch
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import pickle as pkl 
import sys
import argparse
from time import perf_counter_ns
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, './Python_libs')
# load GP function
from random_fields import *

# load GANO model
from importlib import reload
import GANO_model
reload(GANO_model)
from GANO_model import Generator, Discriminator

# import utils
from dataUtils_3C import SeisData
import os
import timeit
from mpi4py import MPI

# adjust the layout of jupyternoteook
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# GANO model parameters
ndim = 6000      # dimension of 1D time history
npad = 400       # pad at the end, to guarantee the length of data is the power of 2 (efficient FFT)
width= 32        # lift the dimension of input from 3 -> width
lr = 1e-4        # learning rate

# Training parameters
epochs = 30      # training epochs
λ_grad = 10.0    # penatly factor
n_critic = 10    # train D n_critic times before train G
batch_size = 32  # decrease batch_size if cuda is out of memory

config_d = {
    
    'data_file': '/scratch/10845/kaichengz/data/vel_100hz_all.npy',       # full dataset, shape [N, 3, ndim]
    'attr_file': '/scratch/10845/kaichengz/data/attributes_100hz_all.csv',     # attribute file, contains magnitude, rupture distance, vs30, etc... for each record. 
    'batch_size': batch_size,

    'frac_train': 0.9,                                              # fraction of training
    'condv_names': ['magnitude','rrup', 'vs30', 'tectonic_value'],  # name of conditional variables
    'condv_min_max' : [(4.0, 8.0), (0, 300), (100, 1100), (0,1)]    # [min, max] for each conditional variable

}

parser = argparse.ArgumentParser(description='Train GANO with optional DDP')
parser.add_argument('--epochs', type=int, default=epochs)
parser.add_argument('--batch-size', type=int, default=batch_size)
parser.add_argument('--n-critic', type=int, default=n_critic)
parser.add_argument('--lr', type=float, default=lr)
parser.add_argument('--master', type=str)
parser.add_argument('--job-id', type=int, default=10000)
args = parser.parse_args()
master_hostname = args.master
job_id = args.job_id

# load the train and val indexes, guarantee reproductivity
ix_train = np.load('./kik_net_data/index_train_100hz_all.npy', allow_pickle=True)
ix_val = np.load('./kik_net_data/index_eval_100hz_all.npy', allow_pickle=True)
# ix_train = index[0]                                                 # index of training dataset                         
# ix_val = index[1]                                                   # index of validation dataset


# run this part if you don't have the index file
# Shuffle the data and save the index.

# Ntot = len(pd.read_csv(config_d['attr_file']))

# frac = config_d['frac_train']
# Nbatch = config_d['batch_size']

# # get all indexes
# ix_all = np.arange(Ntot)
# # get training indexes
# Nsel = int(Ntot*frac)

# ix_train = np.random.choice(ix_all, size=Nsel, replace=False)
# ix_train.sort()
# # get validation indexes
# ix_val = np.setdiff1d(ix_all, ix_train, assume_unique=True)
# ix_val.sort()

# index = []
# index.append(ix_train)
# index.append(ix_val)
# np.save('./kik_net_data/index_train_100hz_all.npy', ix_train)
# np.save('./kik_net_data/index_eval_100hz_all.npy', ix_val)
# exit(0)

epochs = args.epochs
batch_size = args.batch_size
n_critic = args.n_critic
lr = args.lr
config_d['batch_size'] = batch_size

world_size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
print(torch.cuda.is_available(), torch.cuda.device_count())
local_rank = rank % torch.cuda.device_count()
is_distributed = world_size > 1
print(f"world_size {world_size}; rank {rank}")

if is_distributed:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    os.environ["MASTER_ADDR"] = master_hostname
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

if torch.cuda.is_available():
    if is_distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

is_main_process = (rank == 0)
grf = GaussianRF_idct(1, (ndim + npad), alpha=1.5, tau=1.0, cal1d=True, device=device) # 1D GRF (GP)


# data loader
sdat_train = SeisData(config_d['data_file'], config_d['attr_file'],config_d['condv_names'], config_d['condv_min_max'], batch_size=config_d['batch_size'], isel=ix_train)
if is_main_process:
    print('total Train:', sdat_train.get_Ntrain())

sdat_val = SeisData(config_d['data_file'], config_d['attr_file'],config_d['condv_names'], config_d['condv_min_max'], batch_size= config_d['batch_size'], isel=ix_val)
if is_main_process:
    print('total Validation:', sdat_val.get_Ntrain())
    # get random samples [batch, 3, dimension], normalized log10_PGA [batch, 3], conditonal variables[[batch], ..., [batch]] 
    (wfs, log10_PGA, cvs) = sdat_val.get_rand_batch()
    print('shape wfs:', wfs.shape)
    print('shape log10_PGA:', log10_PGA.shape)

train_sampler = DistributedSampler(
    sdat_train, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
) if is_distributed else None
val_sampler = DistributedSampler(
    sdat_val, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True
) if is_distributed else None

train_loader = DataLoader(
    sdat_train,
    batch_size=batch_size,
    sampler=train_sampler,
    shuffle=(train_sampler is None),
    drop_last=True,
)
val_loader = DataLoader(
    sdat_val,
    batch_size=batch_size,
    sampler=val_sampler,
    shuffle=(val_sampler is None),
    drop_last=True,
)

n_train_tot = len(train_loader)


D = Discriminator(6+4, width, ndim=ndim,pad=npad).to(device)    # 6 (3 waveforms+3 PGAs) + 4 (4 conditional variables)
G = Generator(1+4, width, ndim=ndim, pad=npad, training=True).to(device)       # 1 (GP) + 4 (4 conditional variables)

if is_distributed:
    D = DDP(D, device_ids=[local_rank], output_device=local_rank)
    G = DDP(G, device_ids=[local_rank], output_device=local_rank)

nn_params = sum(p.numel() for p in D.parameters() if p.requires_grad)
nn_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
if is_main_process:
    print("Number discriminator parameters: ", nn_params)
    print("Number generator parameters: ", nn_params)


G_optim = torch.optim.Adam(G.parameters(), lr=lr , weight_decay=1e-4)               # optimizer
D_optim = torch.optim.Adam(D.parameters(), lr=lr , weight_decay=1e-4)
G_scheduler = torch.optim.lr_scheduler.StepLR(G_optim, step_size=5, gamma=0.8)      # step learnig rate
D_scheduler = torch.optim.lr_scheduler.StepLR(D_optim, step_size=5, gamma=0.8)

D.train()
G.train()

if is_main_process:
    if not os.path.exists(f"./saved_models/s{job_id}"):
        os.makedirs(f"./saved_models/s{job_id}")
    if not os.path.exists(f"./plots/s{job_id}"):
        os.makedirs(f"./plots/s{job_id}")

def calculate_gradient_penalty(model, real_images, fake_images, label,device):
    """Calculates the gradient penalty loss for WGAN GP"""

    alpha = torch.randn((real_images.size(0), 1, 1), device=device)
    interpolates_wfs = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)
    
    #print(interpolates_wfs.shape, interpolates_lcn.shape)
    model_interpolates = model(interpolates_wfs, label)
    grad_outputs = torch.ones(model_interpolates.size(), device=device, requires_grad=False)

    # Get gradient w.r.t. interpolates
    grad_wf = torch.autograd.grad(
        outputs=model_interpolates,
        inputs=interpolates_wfs,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    #gradients = torch.cat([grad_wf, grad_cn,], 1)
    gradients = grad_wf
    
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1.0/ndim) ** 2)    

    return gradient_penalty


def _to_device_batch(x, log10_PGA, cvs):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if not torch.is_tensor(log10_PGA):
        log10_PGA = torch.tensor(log10_PGA)
    if not torch.is_tensor(cvs):
        cvs = torch.tensor(cvs)

    x = x.float()
    log10_PGA = log10_PGA.float()
    if log10_PGA.dim() == 2:
        log10_PGA = log10_PGA.unsqueeze(2)
    label = cvs.float().unsqueeze(1).to(device)  # [batch, 1, 4]
    return x, log10_PGA, label


def train_WGANO(D, G, epochs, D_optim, G_optim, scheduler=None):
    # record the loss information
    losses_D = np.zeros(epochs)
    losses_G = np.zeros(epochs)
    losses_G_val = np.zeros(epochs)
    losses_W = np.zeros(epochs)
    if is_main_process:
        tbef = perf_counter_ns()
    for i in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(i)
        if val_sampler is not None:
            val_sampler.set_epoch(i)

        loss_D = 0.0
        loss_G = 0.0
        loss_G_val = 0.0
        loss_W = 0.0
        train_iter = iter(train_loader)
        val_iter = iter(val_loader)

        for j in range(n_train_tot):
            for k in range(n_critic):
                D_optim.zero_grad()

                try:
                    (x, log10_PGA, cvs) = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    (x, log10_PGA, cvs) = next(train_iter)
                x, log10_PGA, label = _to_device_batch(x, log10_PGA, cvs)
                x = F.pad(x, [0, npad]).to(device)
                
                x_syn = G(grf.sample(x.shape[0]).to(device), label)
                #print("x_syn shape:{}".format(x_syn.shape))
                
                # wasserstein regularizaiton
                log10_PGA = log10_PGA.repeat(1, 1, (ndim+npad)).to(device)
                x = torch.cat([x, log10_PGA], dim=1)
                # if is_main_process: 
                #     print("###", x_syn[:6], label[:6], D(x, label), D(x_syn, label))
                
                W_loss = -torch.mean(D(x, label)) + torch.mean(D(x_syn, label))
                gradient_penalty = calculate_gradient_penalty(D, x, x_syn, label, device)
                #gradient_penalty = 0

                loss = W_loss + λ_grad * gradient_penalty 
                loss.backward()
                
                loss_D += loss.item()
                loss_W += W_loss.item()

                D_optim.step()
            
            G_optim.zero_grad()
            # train discriminator every n_critic times before updating the generator
            try:
                (x, _, cvs) = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                (x, _, cvs) = next(train_iter)
            x, _, label = _to_device_batch(x, torch.zeros((x.shape[0], 3)), cvs)
            x = x.to(device)

            x_syn = G(grf.sample(x.shape[0]).to(device), label)

            # if is_main_process:
            #     print("###", x_syn[:6], label[:6], D(x_syn, label)) 

            loss = -torch.mean(D(x_syn, label))
            loss.backward()
            loss_G += loss.item()

            G_optim.step()
            
            # Store validation information
            with torch.no_grad():
                if is_main_process:
                    tnow = perf_counter_ns()
                    tdif = tnow - tbef
                    tbef = tnow
                    print("epoch:[{} / {}] batch:[{} / {}] loss_G:{:.4f}  time {}s".format(i, epochs, j, n_train_tot, loss.item(), tdif / 1000000000))   
                
                # save training loss and validation loss 
                try:
                    (x, _, cvs) = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    (x, _, cvs) = next(val_iter)
                x, _, label = _to_device_batch(x, torch.zeros((x.shape[0], 3)), cvs)
                x = x.to(device)

                x_syn = G(grf.sample(x.shape[0]).to(device), label)

                loss = -torch.mean(D(x_syn, label))
                loss_G_val += loss.item()
                
                if j % 200 == 0 and is_main_process:
                    # check the label. 
                    cvs_np = cvs.detach().cpu().numpy() if torch.is_tensor(cvs) else cvs
                    mag = sdat_train.to_real(cvs_np[0][0], 'magnitude')
                    rrup = sdat_train.to_real(cvs_np[0][1], 'rrup')
                    vs30 = sdat_train.to_real(cvs_np[0][2], 'vs30')
                    tectonic_value = sdat_train.to_real(cvs_np[0][3],'tectonic_value')
                    if tectonic_value == 0.0:
                        tectonic_type = 'Subduction'
                    else:
                        tectonic_type = 'Shallow crustal'
                    fig, ax = plt.subplots(1, 1, figsize=(16,8), tight_layout=True)
                    ax.plot(x_syn[0,0,:].squeeze().detach().cpu().numpy())
                    ax.set_title('M {} , {} km, $Vs_{{30}}$={}m/s, event= {}'.format(mag, rrup, vs30, tectonic_type), fontsize=16)
                    plt.savefig(f"./plots/s{job_id}/epoch{i}_it{j}_GANO")
                    plt.close(fig)

        if is_distributed:
            stats_t = torch.tensor([loss_D, loss_G, loss_G_val, loss_W], device=device)
            dist.all_reduce(stats_t, op=dist.ReduceOp.SUM)
            stats_t /= world_size
            loss_D, loss_G, loss_G_val, loss_W = stats_t.tolist()

        losses_D[i] = loss_D / batch_size
        losses_G[i] = loss_G / batch_size
        losses_G_val[i] = loss_G_val / batch_size
        losses_W[i] = loss_W / batch_size
        
        D_scheduler.step()
        G_scheduler.step()
        if (i+1) % 10 == 0 and is_main_process: #save the model every 10 epochs
            g_state = G.module.state_dict() if isinstance(G, DDP) else G.state_dict()
            torch.save(g_state, f"./saved_models/s{job_id}/G_{i+1}_GANO.pt")
        
    return losses_D, losses_G, losses_G_val, losses_W

# create folder is not exist
folder = "GANO_kik_net_training"
if is_main_process:
    if not os.path.exists(f"./saved_models/{folder}"):
        os.makedirs(f"./saved_models/{folder}")
    if not os.path.exists(f"./plots/{folder}"):
        os.makedirs(f"./plots/{folder}")

start = timeit.default_timer() # track the time for training
losses_D, losses_G, losses_G_val, losses_W = train_WGANO(D, G, epochs, D_optim, G_optim)
stop = timeit.default_timer() 
# print time loss
if is_main_process:
    print(stop - start)

if is_main_process:
    plt.figure(figsize=(24,4))
    plt.subplot(1,4,1)
    plt.plot(losses_D[10:])
    plt.subplot(1,4,2)
    plt.plot(losses_G[10:])
    plt.subplot(1,4,3)
    plt.plot(losses_G_val[10:])
    plt.subplot(1,4,4)
    plt.plot(losses_W[10:])

    losses_all = pd.DataFrame()

    losses_all['losses_D'] = losses_D
    losses_all['losses_G'] = losses_G
    losses_all['losses_G_val'] = losses_G_val
    losses_all['losses_W'] = losses_W

    losses_all.to_csv(f"./losses_GANO.s{job_id}.npy")

if is_distributed:
    dist.destroy_process_group()
