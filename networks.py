
from torch import nn
from utils import signal_composiation
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import math
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass


class SeismoDis_ACGAN_bilinear_real_dist(nn.Module):

    def __init__(self, input_dim, gen_length, params):
        super().__init__()
        self.gan_type = params['gan_type']
        self.dim = params['dim']
        self.kernel_size = params['kernel_size']
        self.pad_type = params['pad_type']
        self.input_dim = input_dim
        self.threshold = params.get('threshold', 0.5)
        numb_blocks = 10
        padding = (self.kernel_size - 1) // 2 - 1

        # 提取特征的堆叠残差块
        blocks = [ResBlk(self.input_dim, self.dim, self.kernel_size, padding, norm=True, downsample=True)]
        blocks += [ResBlk(self.dim, self.dim, self.kernel_size, padding, norm=True, downsample=True)
                   for _ in range(numb_blocks - 2)]
        self.feature_extract = nn.Sequential(*blocks)

        # 输出长度用于线性层计算
        output_dims = int(gen_length / 2 ** (numb_blocks - 1)) * self.dim
        self.linear_dis = nn.Sequential(
            LinearBlock(output_dims, output_dims // 2, norm='none', activation='lrelu'),
            LinearBlock(output_dims // 2, 1, norm='none', activation='none')
        )
        self.linear_aux = nn.Sequential(
            LinearBlock(output_dims, output_dims // 2, norm='none', activation='lrelu'),
            LinearBlock(output_dims // 2, 1, norm='none', activation='none')
        )

    def forward(self, x):
        x = self.feature_extract(x)
        x = torch.flatten(x, start_dim=1)
        return self.linear_dis(x).squeeze(-1), self.linear_aux(x).squeeze(-1)

    def embeding(self, x):
        return self.feature_extract(x)

    def calc_dis_loss(self, input_fake, input_real, labels, gradient_penalty, acc):
        input_real.requires_grad_()
        out_fake, aux_fake = self.forward(input_fake.detach())
        out_real, aux_real = self.forward(input_real)

        if self.gan_type == 'lsgan':
            loss = torch.mean((out_fake - 0) ** 2) + torch.mean((out_real - 1) ** 2)
        elif self.gan_type == 'nsgan':
            loss = F.binary_cross_entropy_with_logits(out_fake, torch.zeros_like(out_fake)) + \
                   F.binary_cross_entropy_with_logits(out_real, torch.ones_like(out_real))
        elif self.gan_type == 'wgan':
            loss = out_fake.mean() - out_real.mean()
        elif self.gan_type == 'wgan-gp':
            loss = out_fake.mean() - out_real.mean() + gradient_penalty
        elif self.gan_type == 'acgan':
            loss_adv = 0.5 * (F.binary_cross_entropy_with_logits(out_fake, torch.zeros_like(out_fake)) +
                              F.binary_cross_entropy_with_logits(out_real, torch.ones_like(out_real)))
            aux_loss = 0.5 * (F.l1_loss(aux_fake, labels) + F.l1_loss(aux_real, labels))
            loss = loss_adv + aux_loss
        else:
            raise ValueError(f"Unsupported GAN type: {self.gan_type}")

        if acc:
            fake_dis_acc = (torch.sigmoid(out_fake).detach().cpu().numpy().round() == 0).mean()
            real_dis_acc = (torch.sigmoid(out_real).detach().cpu().numpy().round() == 1).mean()
        else:
            fake_dis_acc = real_dis_acc = None

        return loss, aux_loss if self.gan_type == 'acgan' else torch.tensor(0.0), fake_dis_acc, real_dis_acc


class SeisGen_ACGAN_real_dist(nn.Module):

    def __init__(self, gen_length, params):
        super().__init__()
        self.start_dim = params['start_dim']
        self.cha = params['dim']
        in_dim = params['noise_length'] + 1  # 1 for condition
        kernel_size = params['kernel_size']
        padding = (kernel_size - 2) // 2

        self.linear = nn.Sequential(
            nn.Linear(in_dim, self.start_dim * self.cha),
            nn.BatchNorm1d(self.start_dim * self.cha),
            nn.ReLU()
        )

        layers = []
        dims = [self.cha, self.cha // 2, self.cha // 4, self.cha // 8, self.cha // 16, self.cha // 32]
        for i in range(len(dims) - 1):
            layers.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU()
            ))

        self.upsample_net = nn.Sequential(*layers)
        self.to_waveform = nn.Conv1d(dims[-1], 3, kernel_size=1)

    def forward(self, noise, cond):
        if cond.dim() == 1:
            cond = cond.unsqueeze(1)
        x = torch.cat([noise, cond], dim=1)
        x = self.linear(x)
        x = x.view(-1, self.cha, self.start_dim)
        x = self.upsample_net(x)
        return torch.sigmoid(self.to_waveform(x))



##################################################################################
# Basic Blocks
##################################################################################
import math

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, actv=nn.LeakyReLU(0.2),
                 norm=False, downsample=False):
        super().__init__()
        padding = (padding, padding)
        self.actv = actv
        self.normalize = norm
        self.kernel_size = kernel_size
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

        self.pad = nn.ReflectionPad1d(padding)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv1d(dim_in, dim_in, self.kernel_size, 1, 1)
        self.conv2 = nn.Conv1d(dim_in, dim_out, self.kernel_size, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm1d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm1d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool1d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(self.pad(x))
        if self.downsample:
            x = F.avg_pool1d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(self.pad(x))
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

class dis_Conv_label(nn.Module):
    def __init__(self, input_dim ,output_dim, linear_len, kernel_size, stride,
                 padding=0, norm='in', activation='relu', pad_type='zero'):
        super(dis_Conv_label, self).__init__()

        self.conv = Conv1dBlock(input_dim, output_dim, kernel_size, stride, padding, norm=norm, activation=activation,
                                                  pad_type=pad_type)
        self.linear = LinearBlock(1, linear_len, norm='bn', activation='relu')
        self.conv_label = nn.Sequential(Conv1dBlock(1, 16, kernel_size, 1, padding=kernel_size//2 - 1, norm='bn'))

    def forward(self, x, dist):
        x = self.conv(x)
        dist = self.linear(dist.squeeze(1).float()).unsqueeze(1)
        label = self.conv_label(dist.float())
        return torch.cat((x, label), 1)

class dis_Conv_label_emb(nn.Module):
    def __init__(self, input_dim ,output_dim, linear_len, kernel_size, stride,
                 padding=0, norm='in', activation='relu', pad_type='zero'):
        super(dis_Conv_label_emb, self).__init__()

        self.conv = Conv1dBlock(input_dim, output_dim, kernel_size, stride, padding, norm=norm, activation=activation,
                                                  pad_type=pad_type)
        #self.linear = LinearBlock(1, linear_len, norm='bn', activation='relu')
        self.emb = nn.Embedding(12, linear_len)
        self.conv_label = nn.Sequential(Conv1dBlock(1, output_dim//2, kernel_size, 1, padding=kernel_size//2 - 1, norm='bn'),
                                        Conv1dBlock(output_dim//2, output_dim//2, kernel_size, 1, padding=kernel_size//2 - 1, norm='bn'))

    def forward(self, x, dist):
        x = self.conv(x)
        dist = self.emb(dist)
        label = self.conv_label(dist)
        return torch.cat((x, label), 1)
##################################################################################
# Normalization layers
##################################################################################


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

def r1_reg(d_out, x_in):
    """R1 gradient penalty regularizer for real images"""
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert grad_dout2.size() == x_in.size()
    return 0.5 * grad_dout2.view(x_in.size(0), -1).sum(1).mean(0)

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input_):
        return input_/torch.sqrt(torch.mean(input_**2, dim=1, keepdim=True)+1e-8)