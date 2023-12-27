
import argparse
import math
import random
import os
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
from util import data_sampler, requires_grad, accumulate, sample_data, d_logistic_loss, d_r1_loss, g_nonsaturating_loss, g_path_regularize, make_noise, mixing_noise, set_grad_none

# from model.stylegan.dataset import MultiResolutionDataset
# from model.stylegan.distributed import (
#     get_rank,
#     synchronize,
#     reduce_loss_dict,
#     reduce_sum,
#     get_world_size,
# )
from model.stylegan.non_leaking import augment, AdaptiveAugment
from model.stylegan.model import Generator, Discriminator


class TrainOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Train StyleGAN")
        self.parser.add_argument("--batch", type=int, default=16, help="batch sizes for each gpus")
        self.parser.add_argument("--style", type=str, default='soldier', help="style type")

    def parse(self):
        self.opt = self.parser.parse_args()     
        args = vars(self.opt)
        if self.opt.local_rank == 0:
            print('Load options')
            for name, value in sorted(args.items()):
                print('%s: %s' % (str(name), str(value)))
        return self.opt

#if arch == 'stylegan2':
    #from model.stylegan.model import Generator, Discriminator

#elif arch == 'swagan':
    #from swagan import Generator, Discriminator


def inference():
    # styleGAN2 불러오기
    generator = Generator(
        size, latent, n_mlp, channel_multiplier=channel_multiplier
    ).to(device)
    discriminator = Discriminator(
        size, channel_multiplier=channel_multiplier
    ).to(device)
    g_ema = Generator(
        size, latent, n_mlp, channel_multiplier=channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = g_reg_every / (g_reg_every + 1)
    d_reg_ratio = d_reg_every / (d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    # nowar style에 대해 fine-tuned된 StyleGAN2을 불러오기
    model_name = 'finetune-000200.pt'
    generator_prime = Generator(1024, 512, 8, 2).to(device)
    generator_prime.eval()
    ckpt = torch.load(os.path.join(model_path, style, model_name))
    generator_prime.load_state_dict(ckpt["g_ema"])

    # ffhq pretrained StyleGAN2을 불러오기
    generator = Generator(1024, 512, 8, 2).to(device)
    generator.eval()
    ckpt = torch.load(os.path.join(model_path, 'stylegan2-ffhq-config-f.pt'))
    generator.load_state_dict(ckpt["g_ema"])
    noises_single = generator.make_noise()

    # encoder 불러오기
    encoder_path = os.path.join(model_path, 'encoder.pt')
    ckpt = torch.load(encoder_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = encoder_path
    from argparse import Namespace
    opts = Namespace(**opts)
    # encoder를 eval 모드로 바꾸기
    from model.encoder.psp import pSp
    encoder = pSp(opts)
    encoder.eval()
    encoder.to(device)

    """# styleGAN 실행"""

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    datapath = os.path.join(data_path, 'test/')
    files = os.listdir(datapath)[1:]
    files.sort()

    if batch==1:
        imgs = []
        imgs.append(transform(Image.open(os.path.join(datapath, files[0])).convert("RGB")))
        imgs = torch.stack(imgs, 0).to(device)

        with torch.no_grad():
            # reconstructed face g(z^+_e) and extrinsic style code z^+_e
            img_rec, latent_e = encoder(imgs, randomize_noise=False, return_latents=True, z_plus_latent=True)

        # for j in range(imgs.shape[0]):
        #     dict2[batchfiles[j]] = latent_e[j:j+1].cpu().numpy()

        noises = []
        for noise in noises_single:
            noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())
        for noise in noises:
            noise.requires_grad = True

        # z^+ to be optimized in Eq. (1)
        latent = latent_e.detach().clone()
        latent.requires_grad = True

        latent_n = latent
        img_gen, _ = generator_prime([latent_n], input_is_latent=False, noise=noises, z_plus_latent=True)

        with torch.no_grad():
            img_gen, _ = generator_prime([latent_n], input_is_latent=False, noise=noises, z_plus_latent=True)
            # sample = F.interpolate(sample,256)
            utils.save_image(
                img_gen,
                os.path.join(datapath, "result", f"0sample_{style}_{model_name}.jpg"),
                nrow=int(batch ** 0.5),
                normalize=True,
                #range=(-1, 1),
            )
            print(f"{batchfiles} are done")
    else:
        for ii in range(0, len(files)-1, batch):
            batchfiles = files[ii:ii+batch]
            imgs = []
            for file in batchfiles:
                img = transform(Image.open(os.path.join(datapath, file)).convert("RGB"))
                imgs.append(img)
            imgs = torch.stack(imgs, 0).to(device)

            with torch.no_grad():
                # reconstructed face g(z^+_e) and extrinsic style code z^+_e
                img_rec, latent_e = encoder(imgs, randomize_noise=False, return_latents=True, z_plus_latent=True)

            # for j in range(imgs.shape[0]):
            #     dict2[batchfiles[j]] = latent_e[j:j+1].cpu().numpy()

            noises = []
            for noise in noises_single:
                noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())
            for noise in noises:
                noise.requires_grad = True

            # z^+ to be optimized in Eq. (1)
            latent = latent_e.detach().clone()
            latent.requires_grad = True

            latent_n = latent
            with torch.no_grad():
                img_gen, _ = generator_prime([latent_n], input_is_latent=False, noise=noises, z_plus_latent=True)
                # sample = F.interpolate(sample,256)
                utils.save_image(
                    img_gen,
                    os.path.join(datapath, "result", f"sample_cropped_{style}_{model_name}{ii}.jpg"),
                    nrow=int(batch ** 0.5),
                    normalize=True,
                    #range=(-1, 1),
                )
            print(f"{files[0]} is done")

if __name__ == "__main__":
    
    parser = TrainOptions()
    args = parser.parse()

    if args.style=='soldier':
        style = 'nowar_soldier'
        path = './data/nowar_soldier/lmdb/'
    elif args.style=='victim':
        style = 'nowar_victim'
        path = './data/nowar_victim/lmdb/'

    model_path = './checkpoint/'
    data_path ='./data/'
    local_rank = 0
    size = 1024
    channel_multiplier = 2
    g_reg_every = 4
    d_reg_every = 16
    lr = 0.002
    ckpt_path = './checkpoint/stylegan2-ffhq-config-f.pt'
    iter = 400
    batch = args.batch
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu') # cuda 환경에서만 작동

    latent = 512
    n_mlp = 8
    start_iter = 0

    inference()
