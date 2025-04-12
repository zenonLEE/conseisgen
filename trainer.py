import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad as torch_grad
from networks import SeismoDis_ACGAN_bilinear_real_dist, SeisGen_ACGAN_real_dist
from utils import weights_init, get_model_list, get_scheduler, KLDLoss, Timer
import copy

class Seismo_Trainer_ACGAN_real_dist(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        self.threshold = hyperparameters['dis']['threshold']
        self.gp_weight = hyperparameters['dis']['gp_weight']
        self.gan_type = hyperparameters['dis']['gan_type']
        self.use_ema = hyperparameters.get('use_ema', True)
        self.ema_decay = hyperparameters.get('ema_decay', 0.999)

        self.gen = SeisGen_ACGAN_real_dist(hyperparameters['gen_length'], hyperparameters['gen'])
        self.dis = SeismoDis_ACGAN_bilinear_real_dist(hyperparameters['input_dim'], hyperparameters['gen_length'], hyperparameters['dis'])

        if self.use_ema:
            self.gen_ema = copy.deepcopy(self.gen)

        lr = hyperparameters['lr']
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        weight_decay = hyperparameters['weight_decay']

        self.dis_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.dis.parameters()), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
        self.gen_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.gen.parameters()), lr=lr*10, betas=(beta1, beta2), weight_decay=weight_decay)

        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        self.apply(weights_init(hyperparameters['init']))
        self.dis.apply(weights_init('gaussian'))
        self.gen.apply(weights_init('kaiming'))

    def forward(self, x, dist):
        self.eval()
        fake_sample = self.gen_ema.decode(x, dist) if self.use_ema else self.gen.decode(x, dist)
        self.train()
        return fake_sample

    def gen_update(self, noise, label, acc, ds_loss=False):
        self.gen_opt.zero_grad()
        fake_sample = self.gen(noise, label)
        d_generated, aux_d = self.dis(fake_sample)

        valid = torch.empty_like(d_generated).uniform_(0.9, 1.0).cuda()
        self.g_loss = F.binary_cross_entropy(torch.sigmoid(d_generated), valid)

        aux_loss = torch.mean(torch.abs(label - aux_d))
        self.g_loss += aux_loss

        if ds_loss:
            noise2 = Variable(torch.randn_like(noise)).cuda()
            fake_sample2 = self.gen(noise2, label).detach()
            emb_1 = self.dis.embeding(fake_sample)
            emb_2 = self.dis.embeding(fake_sample2)
            loss_ds = torch.mean(torch.abs(emb_1 - emb_2))
            self.g_loss += -5 * loss_ds
        else:
            loss_ds = torch.tensor(0.0).cuda()

        self.g_loss.backward()
        self.gen_opt.step()

        if self.use_ema:
            self.moving_average(self.gen, self.gen_ema, beta=self.ema_decay)

        fake_dis_acc = (torch.sigmoid(d_generated).detach().cpu().numpy().round() == 1).mean() if acc else 0
        return self.g_loss, aux_loss, fake_dis_acc, loss_ds

    def dis_update(self, noise, real, fake_dist, real_dist, acc):
        fake = self.gen(noise, fake_dist)
        d_fake, _ = self.dis(fake.detach())
        d_real, _ = self.dis(real)

        self.dis_opt.zero_grad()

        if self.gan_type == 'wgan-gp':
            gradient_penalty = self._gradient_penalty(fake, real, fake_dist, real_dist)
            loss_dis, fake_dis_acc, real_dis_acc, real_aux_acc = self.dis.calc_dis_loss(
                fake.detach(), real, fake_dist.detach(), gradient_penalty, acc
            )
        elif self.gan_type == 'hinge':
            loss_fake = torch.mean(F.relu(1. + d_fake))
            loss_real = torch.mean(F.relu(1. - d_real))
            loss_dis = loss_fake + loss_real
            fake_dis_acc = real_dis_acc = real_aux_acc = 0
        elif self.gan_type == 'r1':
            r1 = self.r1_reg(d_real, real)
            loss_dis = torch.mean(F.relu(1. - d_real)) + torch.mean(F.relu(1. + d_fake)) + 10 * r1
            fake_dis_acc = real_dis_acc = real_aux_acc = 0
        else:
            raise ValueError(f"Unsupported GAN type: {self.gan_type}")

        loss_dis.backward()
        self.dis_opt.step()

        return loss_dis, fake_dis_acc, real_dis_acc, real_aux_acc

    def _gradient_penalty(self, generated, real, fake_label, real_label, use_cuda=True):
        batch_size = real.size(0)
        alpha = torch.rand(batch_size, 1, 1).cuda() if use_cuda else torch.rand(batch_size, 1, 1)
        alpha_label = alpha.squeeze()

        interpolated = Variable(alpha * real + (1 - alpha) * generated, requires_grad=True)
        interpolated_label = alpha_label * real_label + (1 - alpha_label) * fake_label

        if use_cuda:
            interpolated, interpolated_label = interpolated.cuda(), interpolated_label.cuda()

        prob_interpolated = self.dis(interpolated, interpolated_label)
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones_like(prob_interpolated),
                               create_graph=True, retain_graph=True)[0].view(batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def r1_reg(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
        return reg

    def sample(self, noise, real, fake_dist):
        self.eval()
        fakes = [self.gen_ema(noise[i:i+1], fake_dist[i:i+1]) if self.use_ema else self.gen(noise[i:i+1], fake_dist[i:i+1]) for i in range(real.size(0))]
        self.train()
        return torch.cat(fakes, 0), real

    def update_learning_rate(self):
        if self.dis_scheduler: self.dis_scheduler.step()
        if self.gen_scheduler: self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        iterations = 0
        gen_path = get_model_list(checkpoint_dir, "gen")
        self.gen.load_state_dict(torch.load(gen_path))
        iterations = int(gen_path[-11:-3])

        dis_path = get_model_list(checkpoint_dir, "dis")
        self.dis.load_state_dict(torch.load(dis_path))

        opt_path = os.path.join(checkpoint_dir, 'optimizer.pt')
        state_dict = torch.load(opt_path)
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])

        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)

        if self.use_ema:
            self.gen_ema = copy.deepcopy(self.gen)

        print(f'Resumed from iteration {iterations}')
        return iterations

    def save(self, snapshot_dir, iterations):
        torch.save(self.gen.state_dict(), os.path.join(snapshot_dir, f'gen_{iterations + 1:08d}.pt'))
        torch.save(self.dis.state_dict(), os.path.join(snapshot_dir, f'dis_{iterations + 1:08d}.pt'))
        torch.save({
            'gen': self.gen_opt.state_dict(),
            'dis': self.dis_opt.state_dict()
        }, os.path.join(snapshot_dir, 'optimizer.pt'))

    def moving_average(self, model, model_test, beta=0.999):
        for param, param_test in zip(model.parameters(), model_test.parameters()):
            param_test.data = torch.lerp(param.data, param_test.data, beta)