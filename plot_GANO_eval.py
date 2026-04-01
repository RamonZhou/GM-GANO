import os
import sys
import glob

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import matplotlib.pyplot as plt
import numpy as np
import torch
from obspy.core.trace import Trace

sys.path.insert(0, './Python_libs')

from GANO_model import Generator
from dataUtils_3C import SeisData
from random_fields import GaussianRF_idct
from tutorial_utils import generate_scen_data, baseline_correction


config = {
    'paths': {
        'data_file': '/scratch/10845/kaichengz/data/vel_100hz.npy',
        'attr_file': '/scratch/10845/kaichengz/data/attributes_100hz.csv',
        'eval_index_file': './kik_net_data/index_eval_100hz.npy',
        'generator_checkpoint': './saved_models/G_30_GANO.628229.pt',
        'output_figure': './plots/GANO_eval_R2.628229.png',
    },
    'model': {
        'ndim': 6000,
        'npad': 400,
        'width': 32,
    },
    'data': {
        'batch_size': 1,
        'condv_names': ['magnitude', 'rrup', 'vs30', 'tectonic_value'],
        'condv_min_max': [(4.0, 8.0), (0, 300), (100, 1100), (0, 1)],
    },
    'processing': {
        'time_step': 0.01,
    },
}


def load_generator_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        checkpoint_candidates = sorted(glob.glob('./saved_models/**/*.pt', recursive=True))
        raise FileNotFoundError(
            'Generator checkpoint not found at "{}". Available .pt files: {}'.format(
                checkpoint_path,
                checkpoint_candidates if checkpoint_candidates else 'none',
            )
        )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    try:
        model.load_state_dict(checkpoint)
    except RuntimeError:
        checkpoint = {
            key.replace('module.', '', 1): value
            for key, value in checkpoint.items()
        }
        model.load_state_dict(checkpoint)


def normalized_velocity_to_raw(wfs_norm, log10_scale_norm, fn_to_real):
    scale_real = np.power(10.0, fn_to_real(log10_scale_norm))
    return wfs_norm * scale_real[:, np.newaxis]


def velocity_to_acceleration_with_trace(wfs_vel, dt):
    wfs_acc = np.zeros_like(wfs_vel, dtype=np.float32)
    for comp in range(3):
        tr = Trace(data=np.asarray(wfs_vel[comp], dtype=np.float32), header={"delta": dt})
        tr.differentiate()
        tr.detrend('demean')
        wfs_acc[comp] = tr.data
    return wfs_acc


def compute_r2(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ss_res = np.sum((x - y) ** 2)
    ss_tot = np.sum((x - np.mean(x)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def validate_required_paths():
    missing_paths = []
    for path_key in ['data_file', 'attr_file', 'eval_index_file', 'generator_checkpoint']:
        path_value = config['paths'][path_key]
        if not os.path.exists(path_value):
            missing_paths.append((path_key, path_value))

    if missing_paths:
        message_lines = ['Missing required input files from config:']
        for path_key, path_value in missing_paths:
            message_lines.append(f'  {path_key}: {path_value}')
        raise FileNotFoundError('\n'.join(message_lines))


def main():
    ndim = config['model']['ndim']
    npad = config['model']['npad']
    width = config['model']['width']
    dt = config['processing']['time_step']
    time = np.arange(ndim) * dt

    validate_required_paths()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ix_val = np.load(config['paths']['eval_index_file'], allow_pickle=True)
    sdat_val = SeisData(
        config['paths']['data_file'],
        config['paths']['attr_file'],
        config['data']['condv_names'],
        config['data']['condv_min_max'],
        batch_size=config['data']['batch_size'],
        isel=ix_val,
    )

    grf = GaussianRF_idct(1, (ndim + npad), alpha=1.5, tau=1.0, cal1d=True, device=device)

    G = Generator(1 + 4, width, pad=npad, ndim=ndim)
    load_generator_checkpoint(G, config['paths']['generator_checkpoint'])
    G.to(device)
    G.eval()

    mx_all = []
    my_all = []

    for idx in range(len(sdat_val)):
        wfs_norm, log10_pga, cvs = sdat_val[idx]

        fake_wfs = generate_scen_data(
            G=G,
            grf=grf,
            v_all=cvs,
            time_step=dt,
            velocity=True,
            one_condition=True,
            n_syn=1,
            ndim=ndim,
            device=device,
        )
        fake_wfs_corrected = baseline_correction(time, fake_wfs)[0]

        gt_vel = normalized_velocity_to_raw(wfs_norm, log10_pga, sdat_val.fn_to_real)
        gt_acc = velocity_to_acceleration_with_trace(gt_vel, dt)

        mx_all.append(np.max(np.abs(gt_acc), axis=1))
        my_all.append(np.max(np.abs(fake_wfs_corrected), axis=1))

    mx_all = np.asarray(mx_all)
    my_all = np.asarray(my_all)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), tight_layout=True)

    for comp in range(3):
        mx = mx_all[:, comp]
        my = my_all[:, comp]
        r2 = compute_r2(mx, my)

        xy_min = min(np.min(mx), np.min(my))
        xy_max = max(np.max(mx), np.max(my))

        axes[comp].scatter(mx, my, s=18, alpha=0.7)
        axes[comp].plot([xy_min, xy_max], [xy_min, xy_max], 'r--', linewidth=1.5)
        axes[comp].set_title(f'Component {comp + 1}')
        axes[comp].set_xlabel('Ground Truth Peak |acc|')
        axes[comp].set_ylabel('GANO Peak |acc|')
        axes[comp].text(
            0.05,
            0.95,
            f'$R^2$ = {r2:.4f}',
            transform=axes[comp].transAxes,
            va='top',
            ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )

    output_dir = os.path.dirname(config['paths']['output_figure'])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig.savefig(config['paths']['output_figure'], dpi=200)
    plt.close(fig)

    print('Saved figure to:', config['paths']['output_figure'])
    for comp in range(3):
        print(f'Component {comp + 1} R^2: {compute_r2(mx_all[:, comp], my_all[:, comp]):.6f}')


if __name__ == '__main__':
    main()
