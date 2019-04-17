import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


class Options:
    batch_size = 256  # number of data in one mini-batch
    learning_rate = 2e-4
    betas = (.5, .999)  # betas for Adam Optimizer
    n_epochs = 50  # total number of training epochs
    result_dir = 'resultsPGGAN'  # test images will be stored in this path


opt = Options()


class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(100, 512, 4, 1, 3, bias=True),  # 4 x 4
            nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=True),  # 4 x 4
            nn.LeakyReLU(.2, inplace=True)
        )

        self.block2 = nn.Sequential(
            nn.Upsample(scale_factor=2),              # 8 x 8
            nn.Conv2d(512, 256, 3, 1, 1, bias=True),  # 8 x 8
            nn.LeakyReLU(.2, inplace=True)
        )

        self.block3 = nn.Sequential(
            nn.Upsample(scale_factor=2),              # 16 x 16
            nn.Conv2d(256, 128, 3, 1, 1, bias=True),  # 16 x 16
            nn.LeakyReLU(.2, inplace=True)
        )

        self.block4 = nn.Sequential(
            nn.Upsample(scale_factor=2),              # 32 x 32
            nn.Conv2d(128, 64, 3, 1, 1, bias=True),   # 32 x 32
            nn.LeakyReLU(.2, inplace=True),
        )

        self.block5 = nn.Sequential(
            nn.Upsample(scale_factor=2),              # 64 x 64
            nn.Conv2d(64, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(32, 1, 1, 1, 0, bias=True)
        )

        self.model = nn.Sequential(
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.block5
        )

    def forward(self, *input, phase=0):
        return self.model[:phase+1](*input)


class Discriminator(nn.Module):

    def __init__(self, mstd_N_group=4):
        super().__init__()

        class Flatten(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, input: torch.Tensor):
                return input.view(input.size(0), -1)

        class MinibatchStandardDeviation(nn.Module):
            def __init__(self, N_group=mstd_N_group):
                """Minibatch will be divided in N_group's"""
                super().__init__()
                self.N_group = N_group
            def forward(self, input: torch.Tensor):
                """
                In : N x C x H x W
                Out : N x (C + 1) x H x W
                """
                y = input
                s = y.size()
                y = y.view(self.N_group, -1, *s[1:])  # GMCHW
                y -= y.mean(dim=0, keepdim=True)  # GMCHW
                y = (y ** 2).mean(dim=0)  # MCHW
                y = y.sqrt()  # MCHW
                y = y.mean(dim=[1, 2, 3], keepdim=True)  # M111
                y = y.repeat(self.N_group, 1, *s[2:])  # N1HW
                return torch.cat((input, y), dim=1)  # N(C+1)HW

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 1, 1, 0, bias=True),
            nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(.2, inplace=True),
            nn.MaxPool2d(2, 2, 0),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(.2, inplace=True),
            nn.MaxPool2d(2, 2, 0),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(.2, inplace=True),
            nn.MaxPool2d(2, 2, 0),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, bias=True),
            nn.LeakyReLU(.2, inplace=True),
            nn.MaxPool2d(2, 2, 0),
        )

        self.block5 = nn.Sequential(
            MinibatchStandardDeviation(mstd_N_group),
            nn.Conv2d(513, 512, 3, 1, 1, bias=True),
            nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(512, 100, 3, 1, 1, bias=True),
            nn.LeakyReLU(.2, inplace=True),
            Flatten(),
            nn.Linear(4 * 4 * 100, 1, bias=True)
        )

        self.model = nn.Sequential(
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.block5
        )

    def forward(self, *input, phase=0):
        return self.model[-(phase+1):](*input)


class PGGAN:

    def __init__(
            self,
            lr=opt.learning_rate, betas=opt.betas
    ):
        self.learning_rate = lr
        self.betas = betas

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.define_nets()
        self.define_criterions()
        self.move_to()
        self.define_optimizers()

    def define_nets(self):
        self.netG = Generator()
        self.netD = Discriminator()

        # no dropouts, no BNs, I think these are not needed.
        self.netG.train()
        self.netD.train()

    def define_criterions(self):
        self.criterionGAN =