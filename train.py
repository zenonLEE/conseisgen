import os
import sys
import shutil
import argparse
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from utils import (
    load_kiknet, normalization, prepare_sub_folder, write_html, write_loss, get_config,
    write_2images, Timer, count_parameters
)
from trainer import Seismo_Trainer_ACGAN_real_dist
import tensorboardX

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/seismo.yaml', help='Path to config file')
    parser.add_argument('--output_path', type=str, default='.', help='Output directory')
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument('--trainer', type=str, default='MUNIT', help="Trainer type: MUNIT|UNIT")
    opts = parser.parse_args()

    # CUDA 设置
    print(f"CUDA Available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}")
    cudnn.benchmark = True
    opts.resume = False

    # 加载配置
    config = get_config(opts.config)
    max_iter = config['max_iter']
    display_size = config['display_size']

    # 初始化模型
    trainer = Seismo_Trainer_ACGAN_real_dist(config).cuda()
    count_parameters(trainer.gen)
    count_parameters(trainer.dis)

    # 加载数据并标准化
    seismic_data_raw, mag = load_kiknet(length=config['gen_length'], low_cut=0.01, highcut=30, label='dis', mag_range=[4, 5])
    seismic_data, mag_ = normalization(seismic_data_raw, mag, 'minmax')
    seismic_data = seismic_data.swapaxes(2, 1).astype(np.float32)
    mag = np.array(mag_)

    print('Check data length:', len(seismic_data), len(mag))

    # 设置日志和输出目录
    model_name = 'acgan_real_dist_kiknet_45'
    log_dir = os.path.join(opts.output_path, "logs", model_name)
    train_writer = tensorboardX.SummaryWriter(log_dir)
    output_directory = os.path.join(opts.output_path, "outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

    iterations = trainer.resume(checkpoint_directory, config) if opts.resume else 0

    # 训练主循环
    while iterations < max_iter:
        acc = (iterations + 1) % config['log_iter'] == 0
        idx = np.random.choice(len(seismic_data), config['batch_size'])
        images_a = torch.tensor(seismic_data[idx]).cuda()
        label = torch.tensor(mag[idx]).cuda()
        dist_label = fake_dist_label = label

        noise = Variable(torch.randn(config['batch_size'], config['gen']['noise_length']).cuda())
        d_loss, aux_loss, fake_dis_acc, real_dis_acc = trainer.dis_update(noise, images_a, fake_dist_label, dist_label, acc)
        g_loss, g_aux_loss, g_fake_acc, g_ds_loss = trainer.gen_update(noise, fake_dist_label, acc, ds_loss=True)
        trainer.update_learning_rate()
        torch.cuda.synchronize()

        if acc:
            print(f"Iteration: {iterations + 1:08d}/{max_iter:08d}; d_loss: {d_loss:.2f}; g_loss: {g_loss:.2f}; aux_loss: {aux_loss:.2f}; g_aux_loss: {g_aux_loss:.2f}; g_ds_loss: {g_ds_loss:.2f}")
            print(f"fake_dis_acc: {fake_dis_acc:.2f}; real_dis_acc: {real_dis_acc:.2f}; all_dis_acc: {(real_dis_acc + fake_dis_acc) * 0.5:.2f}; gen_acc: {g_fake_acc:.2f}")
            write_loss(iterations, trainer, train_writer)

        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                display_noise = Variable(torch.randn(config['display_size'], config['gen']['noise_length']).cuda())
                train_image_outputs = trainer.sample(display_noise, torch.tensor(seismic_data[:config['display_size']]).cuda(), fake_dist_label)
            write_2images(train_image_outputs, display_size, image_directory, f'train_{iterations + 1:08d}')
            write_html(os.path.join(output_directory, "index.html"), iterations + 1, config['image_save_iter'], 'images')

        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                display_noise = Variable(torch.randn(config['display_size'], config['gen']['noise_length']).cuda())
                image_outputs = trainer.sample(display_noise, torch.tensor(seismic_data[:config['display_size']]).cuda(), fake_dist_label)
            write_2images(image_outputs, display_size, image_directory, 'train_current')

        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1

    print("Training completed.")
    sys.exit('Finish training')

if __name__ == '__main__':
    main()
