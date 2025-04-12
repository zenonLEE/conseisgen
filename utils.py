
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler

from prettytable import PrettyTable
import torch
import matplotlib.pyplot as plt
from glob import glob
import torch.nn as nn
import os
import math
from scipy.signal import butter, lfilter
import torchvision.utils as vutils
import yaml
import numpy as np
import torch.nn.init as init
import time

def normalization(data, mag, type):
    if type == 'std':
        data_, mag_ = std_normalization(data, mag)
    elif type == 'tanh':
        data_, mag_ = tanh_normalization(data, mag)
    elif type == 'minmax':
        data_, mag_ = minmax_normalization(data, mag)
    return data_, mag_

def tanh_normalization(seismic_data_, mag):
    seismic_data = []
    mag_ = []
    for i in range(seismic_data_.shape[0]):
        std = np.std(seismic_data_[i], axis=0)
        if std[0]!=0 and std[1]!=0 and std[2]!=0:
            # standard score
            e = seismic_data_[i, :, 0]
            n = seismic_data_[i, :, 1]
            z = seismic_data_[i, :, 2]
            e_hat = (e-np.mean(e)+0.00001)/np.std(e)
            n_hat = (n-np.mean(n)+0.00001)/np.std(n)
            z_hat = (z-np.mean(z)+0.00001)/np.std(z)
            e_minmax = e_hat/np.max(abs(e_hat))
            n_minmax =  n_hat/np.max(abs(n_hat))
            z_minmax =  z_hat/np.max(abs(z_hat))

            seismic_data.append(np.concatenate([e_minmax[np.newaxis, :, np.newaxis],
                                                n_minmax[np.newaxis, :, np.newaxis],
                                                z_minmax[np.newaxis, :, np.newaxis]], axis=2))
            mag_.append(mag[i])

    seismic_data = np.concatenate(seismic_data, axis=0)
    return seismic_data, mag_


def std_normalization(seismic_data_, mag):
    seismic_data = []
    mag_ = []
    for i in range(seismic_data_.shape[0]):
        std = np.std(seismic_data_[i], axis=0)
        if std[0]!=0 and std[1]!=0 and std[2]!=0:
            # standard score
            e = seismic_data_[i, :, 0]
            n = seismic_data_[i, :, 1]
            z = seismic_data_[i, :, 2]
            e_hat = (e-np.mean(e)+0.00001)/np.std(e)
            n_hat = (n-np.mean(n)+0.00001)/np.std(n)
            z_hat = (z-np.mean(z)+0.00001)/np.std(z)

            seismic_data.append(np.concatenate([e_hat[np.newaxis, :, np.newaxis], n_hat[np.newaxis, :, np.newaxis],
                                                z_hat[np.newaxis, :, np.newaxis]], axis=2))
            mag_.append(mag[i])

    seismic_data = np.concatenate(seismic_data, axis=0)
    return seismic_data, mag_

def minmax_normalization(seismic_data_, mag):
    seismic_data = []
    mag_ = []
    for i in range(seismic_data_.shape[0]):
        std = np.std(seismic_data_[i], axis=0)
        if std[0]!=0 and std[1]!=0 and std[2]!=0:
            # standard score
            e = seismic_data_[i, :, 0]
            n = seismic_data_[i, :, 1]
            z = seismic_data_[i, :, 2]
            e_hat = (e-np.mean(e)+0.00001)/np.std(e)
            n_hat = (n-np.mean(n)+0.00001)/np.std(n)
            z_hat = (z-np.mean(z)+0.00001)/np.std(z)
            e_minmax = (e_hat-np.min(e_hat))/(np.max(e_hat)-np.min(e_hat))
            n_minmax = (n_hat - np.min(n_hat))/ (np.max(n_hat) - np.min(n_hat))
            z_minmax = (z_hat - np.min(z_hat))/ (np.max(z_hat) - np.min(z_hat))

            seismic_data.append(np.concatenate([e_minmax[np.newaxis, :, np.newaxis],
                                                n_minmax[np.newaxis, :, np.newaxis],
                                                z_minmax[np.newaxis, :, np.newaxis]], axis=2))
            mag_.append(mag[np.newaxis, :, i])

    seismic_data = np.concatenate(seismic_data, axis=0)
    return seismic_data, np.concatenate(mag_, 0)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def load_kiknet(length, low_cut, highcut, label, mag_range):

    file_dir = 'H:\Weather_Metro_KIK_0820_update'
    npz_files = glob(file_dir+'/*.npz')

    data = []
    dist_ = []
    name = []
    ori_tim = []
    mags = []
    #for i in range(len(npz_files)):
    for i in range(1):
        data_file = np.load(npz_files[i])
        dataA = data_file['ev']
        dist = data_file[label]
        mag = data_file['mag']
        ev_name = data_file['ev_name']
        ori_time = data_file['origin_time']

        for k in range(len(dataA)):
            if mag[k] <mag_range[1] and mag[k]>=mag_range[0] and dist[k]<=120:
                snr = snr_cal(dataA[k])
                if snr >=5.:
                    data_ = dataA[k][np.newaxis, :length, :]
                    dist_.append(dist[k])
                    data.append(data_)
                    name.append(ev_name[k])
                    ori_tim.append(ori_time[k])
                    mags.append(mag[k])

    dataA = np.concatenate(data, axis=0).astype('float32')

    ### band pass filter
    data = []
    for i in range(len(dataA)):
        filtered_data = butter_bandpass_filter(dataA[i], low_cut, highcut, fs=100)
        data.append(np.expand_dims(filtered_data, axis=0))
    data = np.concatenate(data, axis=0).astype('float32')
    return data[:, :length, :], np.concatenate((np.array(dist_)[np.newaxis, :], np.array(mags)[np.newaxis, :]),0)

def snr_cal(data, length=500, margin=20):
    data = data[:1000, 2]
    noise = data[:length-margin]
    signal = data[length+margin:]
    ratio = np.linalg.norm(signal, ord=2, axis=0)/np.linalg.norm(noise, ord=2, axis=0)
    snr = 10*np.log10(ratio)
    return snr

def load_seismic(length, margin, low_cut, label, file_name):
    file_dir = './datasets/{}'.format(file_name)
    data_file = np.load(file_dir)
    dataA = data_file['ev']
    dist = data_file[label]

    p_pick = data_file['p_pick'].astype('int32')
    data = []
    dist_ = []
    for k in range(len(dataA)):
        if p_pick[k]>=margin and dist[k]<= 120.:
            data_ = dataA[k][np.newaxis, int(p_pick[k])-margin:int(p_pick[k])+length-margin, :]
            if len(data_[0, :, 0]) == length:
                dist_.append(dist[k])
                data.append(data_)
    dataA = np.concatenate(data, axis=0).astype('float32')
    dist = np.array(dist_)
    ### high pass filter
    data = []
    for i in range(len(dataA)):
        filtered_data = butter_bandpass_filter(dataA[i], low_cut, fs=100)
        data.append(np.expand_dims(filtered_data, axis=0))
    data = np.concatenate(data, axis=0).astype('float32')
    return data[:, :length, :], dist

def load_seismic_all(length, margin, low_cut, label):
    file_dir = './datasets/dataset_overall_mag.npz'
    data_file = np.load(file_dir)
    dataA = data_file['ev']
    dist = data_file[label]

    p_pick = data_file['p_pick'].astype('int32')
    data = []
    dist_ = []
    for k in range(len(dataA)):
        if p_pick[k]>=margin:
            data_ = dataA[k][np.newaxis, int(p_pick[k])-margin:int(p_pick[k])+length-margin, :]
            if data_.shape[1] == length:
                dist_.append(dist[k])
                data.append(data_)
    dataA = np.concatenate(data, axis=0).astype('float32')
    dist = np.array(dist_)
    ### high pass filter
    data = []
    for i in range(len(dataA)):
        filtered_data = butter_bandpass_filter(dataA[i], low_cut, fs=100)
        data.append(np.expand_dims(filtered_data, axis=0))
    data = np.concatenate(data, axis=0).astype('float32')
    return data[:, :length, :], dist

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y_ = []
    for i in range(3):
        y = lfilter(b, a, data[:, i])
        y_.append(np.expand_dims(y, axis=1))
    y_ = np.concatenate(y_, axis=1)
    return y_


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def KLDLoss(mu, logvar):
    #return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar)
    #return -0.5 * torch.sum(-logvar.exp() - torch.pow(mu, 2) + logvar +1, 1)
    # kld_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    # kld = torch.sum(kld_element).mul_(-0.5)
    # return kld

def eformat(f, prec):
    s = "%.*e"%(prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%d"%(mantissa, int(exp))

def test_evluation_image(images, file_name):
    length = images.shape[2]
    numbs = images.shape[0]
    title = list(np.repeat('fake', numbs//2))+ list(np.repeat('real', numbs//2))
    fig, axes = plt.subplots(nrows=3, ncols=numbs, figsize=(12, 4), sharex=True)

    for i in range(numbs):
        for j in range(3):
            if j == 0:
                axes[j, i].set_title(title[i])
            ti = np.arange(0.0, length/100, step=1 / 100)
            axes[j, i].plot(ti, np.squeeze(images[i, j, :]), linewidth=0.5)
            #axes[j, i].set_ylim(0, 1)
            if j == 2:
                axes[j, i].set_xlabel('Time [sec]')

    plt.tight_layout()
    plt.savefig(file_name)

def __write_images(image_outputs, display_image_num, file_name):
    image_outputs = [images.expand(-1, 3, -1) for images in image_outputs] # expand gray-scale images to 3 channels
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    test_evluation_image(image_tensor.cpu().numpy(), file_name)


def write_2images(image_outputs, display_image_num, image_directory, postfix):
    __write_images(image_outputs, display_image_num, '%s/gen_%s.jpg' % (image_directory, postfix))



def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def write_one_row_html(html_file, iterations, img_filename, all_size):
    html_file.write("<h3>iteration [%d] (%s)</h3>" % (iterations,img_filename.split('/')[-1]))
    html_file.write("""
        <p><a href="%s">
          <img src="%s" style="width:%dpx">
        </a><br>
        <p>
        """ % (img_filename, img_filename, all_size))
    return


def write_html(filename, iterations, image_save_iterations, image_directory, all_size=1536):
    html_file = open(filename, "w")
    html_file.write('''
    <!DOCTYPE html>
    <html>
    <head>
      <title>Experiment name = %s</title>
      <meta http-equiv="refresh" content="30">
    </head>
    <body>
    ''' % os.path.basename(filename))
    html_file.write("<h3>current</h3>")
    write_one_row_html(html_file, iterations, '%s/gen_a2b_train_current.jpg' % (image_directory), all_size)
    write_one_row_html(html_file, iterations, '%s/gen_b2a_train_current.jpg' % (image_directory), all_size)
    for j in range(iterations, image_save_iterations-1, -1):
        if j % image_save_iterations == 0:
            write_one_row_html(html_file, j, '%s/gen_a2b_test_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_b2a_test_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_a2b_train_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_b2a_train_%08d.jpg' % (image_directory, j), all_size)
    html_file.write("</body></html>")
    html_file.close()


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer) \
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


def slerp(val, low, high):
    """
    original: Animating Rotation with Quaternion Curves, Ken Shoemake
    https://arxiv.org/abs/1609.04468
    Code: https://github.com/soumith/dcgan.torch/issues/14, Tom White
    """
    omega = np.arccos(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def get_slerp_interp(nb_latents, nb_interp, z_dim):
    """
    modified from: PyTorch inference for "Progressive Growing of GANs" with CelebA snapshot
    https://github.com/ptrblck/prog_gans_pytorch_inference
    """

    latent_interps = np.empty(shape=(0, z_dim), dtype=np.float32)
    for _ in range(nb_latents):
        low = np.random.randn(z_dim)
        high = np.random.randn(z_dim)  # low + np.random.randn(512) * 0.7
        interp_vals = np.linspace(0, 1, num=nb_interp)
        latent_interp = np.array([slerp(v, low, high) for v in interp_vals],
                                 dtype=np.float32)
        latent_interps = np.vstack((latent_interps, latent_interp))

    return latent_interps[:, :, np.newaxis, np.newaxis]


# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name



def vgg_preprocess(batch):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim = 1)
    batch = torch.cat((b, g, r), dim = 1) # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5 # [-1, 1] -> [0, 255]
    mean = tensortype(batch.data.size()).cuda()
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean)) # subtract mean
    return batch


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler

def signal_composiation(input_signal, threshold):

    low_signal = torch.rfft(input_signal, signal_ndim=2)
    zero = torch.zeros_like(low_signal)
    low_signal = torch.where(low_signal >= threshold, zero, low_signal)
    low_signal = torch.irfft(low_signal, signal_ndim=2)

    high_signal = torch.rfft(input_signal, signal_ndim=2)
    high_signal = torch.where(high_signal < threshold, zero, high_signal)
    high_signal = torch.irfft(high_signal, signal_ndim=2)
    return low_signal, high_signal

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))

