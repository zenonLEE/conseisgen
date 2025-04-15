import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.autograd import Variable
from scipy import signal
from utils import get_config, pytorch03_to_pytorch04
from trainer import Seismo_Trainer_ACGAN_real_dist


def stft_batch(inputs, sample_rate):
    """Batch-wise short-time Fourier transform for 3-channel seismic waveforms."""
    all_Sxx = []
    for i in range(len(inputs)):
        x = inputs[i]
        Sxx_channels = []
        for ch in range(3):
            _, _, Sxx = signal.spectrogram(np.squeeze(x[ch, :]), fs=sample_rate, nperseg=50,
                                           noverlap=25, nfft=256, scaling='density')
            Sxx = (Sxx - np.min(Sxx)) / (np.max(Sxx) - np.min(Sxx))
            Sxx_channels.append(np.expand_dims(Sxx, axis=2))
        Sxx_ = np.concatenate(Sxx_channels, axis=2)
        all_Sxx.append(np.expand_dims(Sxx_, axis=0))
    return _, _, np.concatenate(all_Sxx, axis=0)


def plot_seismic_waveform(images, file_name):
    """Plot a 3-channel waveform into a single image."""
    length = images.shape[1]
    time = np.arange(0.0, length / 100, step=1 / 100)
    fig, axes = plt.subplots(nrows=3, figsize=(4, 4), sharex=True)
    for i in range(3):
        axes[i].plot(time, np.squeeze(images[i, :]), linewidth=0.5)
        if i == 2:
            axes[i].set_xlabel('Time [sec]')
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


# -----------------------------
# Argument Parser
# -----------------------------
parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, default=f'./confgis/sesmo.yaml', help="Path to the config file")
parser.add_argument('--display_size', type=int, default=1000, help="Number of waveforms to generate")
parser.add_argument('--output_folder', type=str, default='test_results', help="Folder to save results")
parser.add_argument('--checkpoint', type=str, default=f'outputs/checkpoints/gen_00270000.pt', help="Path to model checkpoint")
parser.add_argument('--seed', type=int, default=10, help="Random seed")
parser.add_argument('--output_path', type=str, default='.', help="Root output path")
opts = parser.parse_args()

# -----------------------------
# Output folder setup
# -----------------------------
opts.output_folder = os.path.join(opts.output_folder, opts.checkpoint.split('/')[1])
os.makedirs(opts.output_folder, exist_ok=True)

# -----------------------------
# Seed
# -----------------------------
torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

# -----------------------------
# Load model and config
# -----------------------------
config = get_config(opts.config)
trainer = Seismo_Trainer_v4(config)
state_dict = torch.load(opts.checkpoint)
trainer.gen.load_state_dict(state_dict)
trainer.cuda()
trainer.eval()

decode = trainer.gen.decode

# -----------------------------
# Inference & Visualization
# -----------------------------
with torch.no_grad():
    noise = Variable(torch.randn(opts.display_size, 1, config['noise_length']).cuda())
    fake = decode(noise).cpu().numpy()

    print('Saving waveform data to .npz')
    np.savez(os.path.join(opts.output_folder, 'data.npz'), ev=fake)

    print('Plotting mean spectrogram...')
    f, t, spec = stft_batch(fake, sample_rate=100)
    mean_spec = np.mean(spec, axis=0)

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    for i in range(3):
        axes[i].pcolor(t, f, np.squeeze(mean_spec[:, :, i]))
        axes[i].set_ylim([0, 50])
        if i == 1:
            axes[i].set_xlabel('Frequency [Hz]')
        if i == 2:
            axes[i].set_xlabel('Time [sec]')
    plt.tight_layout()
    plt.savefig(os.path.join(opts.output_folder, 'mean_spec.png'))
    plt.close()

    print('Saving waveform plots...')
    for i in range(fake.shape[0]):
        plot_seismic_waveform(fake[i], os.path.join(opts.output_folder, f'gen_{i}.png'))
