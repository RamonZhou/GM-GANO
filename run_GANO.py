import torch
import numpy as np
import pylab as plt
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pickle as pkl
import pandas as pd
import os
import sys

sys.path.insert(0, './Python_libs')
# load trained_model
import GANO_model
from GANO_model import Generator
# load GP function
from random_fields import *
from obspy.core.trace import Trace 

# load utils
#from imp import reload
#import tutorial_utils
#reload(tutorial_utils)
from tutorial_utils import *

from dataUtils_3C import SeisData

# adjust the layout of jupyternoteook
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

config = {
    
    'data_file': '/scratch/10845/kaichengz/data/vel_100hz.npy',       # full dataset, shape [N, 3, ndim]
    'attr_file': '/scratch/10845/kaichengz/data/attributes_100hz.csv',     # attribute file, contains magnitude, rupture distance, vs30, etc... for each record. 

    'frac_train': 0.9,                                              # fraction of training
    'condv_names': ['magnitude','rrup', 'vs30', 'tectonic_value'],  # name of conditional variables
    'condv_min_max' : [(4.0, 8.0), (0, 300), (100, 1100), (0,1)]    # [min, max] for each conditional variable

}

# load eval data
ix_val = np.load('./kik_net_data/index_eval_100hz.npy', allow_pickle=True)
sdat_val = SeisData(config['data_file'], config['attr_file'],config['condv_names'], config['condv_min_max'], batch_size=1, isel=ix_val)

ndim = 6000        # dimension of 1D time history
npad = 400         # pad at the end, to guarantee the length of data is the power of 2 (efficient FFT)
width= 32          # lift the dimension of input

# load the model to GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 1D GRF (GP) functoin
grf = GaussianRF_idct(1,(ndim + npad), alpha=1.5, tau=1.0, cal1d=True, device=device)

# Generator
G = Generator(1+4, width, pad=npad, ndim=ndim)
nn_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
print("Number of generator parameters: ", nn_params)

G_path = './saved_models/G_30_GANO.628229.pt'
checkpoint = torch.load(G_path, map_location='cpu') #first load model into cpu, then move to GPU 
G.load_state_dict(checkpoint)
G.to(device)


(wfs, log10_PGA, cvs) = sdat_val.get_rand_batch()
label = np.asarray(cvs)
label = torch.from_numpy(label).permute(1, 2, 0).float()[0][0]
print(label.shape)

v_all = label
v_names = revert_attributes(config=config, v_all=v_all)
print(label, v_names)

# v_names = { 'magnitude':4.5,'rrup':100,'vs30':300,'tectonic_value':0}
# v_all = convert_attributes(config=config, v_names=v_names)

time_step = 0.01   # sampling frequency is 100Hz
times = np.arange(ndim) * time_step # 60s 

# fake_wfs_scen shape [100, 3, 6000], generate 100 waveforms by default
fake_wfs_scen = generate_scen_data(G=G, grf=grf, v_all=v_all, time_step=time_step, velocity=True, one_condition=True, device=device)

## apply the baseline correction
fake_wfs_scen_corrected = baseline_correction(times, fake_wfs_scen)

plot_one_example(fake_wfs_scen_corrected[0], v_names, "result_GANO.png")

gt = wfs[0].copy()
gt_tr = [Trace(data=gt[i], header={"delta":0.01}) for i in range(3)]
gt_acc = []
for i in range(3):
    gt_tr[i].differentiate()
    gt_tr[i].detrend('demean')
    gt_acc.append(gt_tr[i].data)
plot_one_example(np.asarray(gt_acc, dtype=np.float32), v_names, "result_GT.png")

