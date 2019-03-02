import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt


class Options:
    batch_size = 64
    learning_rate = 2e-4
    betas = (.5, .999)
    n_epochs = 100


opt = Options()


class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(128, 256),
            nn.Linear(256, 784)
        )

    def forward(self, *input):
        return self.model(*input)


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.Linear(256, 1)
        )

    def forward(self, *input):
        return self.model(*input)


def define_G():
    return Generator()


def define_D():
    return Discriminator()


class GAN:

    def __init__(self, lr=2e-4, betas=(.5, .999), n_epochs=100):
        self.learning_rate = lr
        self.betas = betas
        self.n_epochs = n_epochs

        self.true_label = torch.tensor(1.)
        self.false_label = torch.tensor(0.)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.define_nets()
        self.define_criterions()
        self.define_optimizers()

    def define_nets(self):
        self.netG = define_G()
        self.netD = define_D()

        self.G_params = self.netG.parameters()
        self.D_params = self.netD.parameters()

    def define_criterions(self):
        self.criterionGAN = torch.nn.BCELoss()

    def move_to(self):
        self.netG         = self.netG.to(self.device)
        self.netD         = self.netD.to(self.device)
        self.criterionGAN = self.criterionGAN.to(self.device)
        self.true_label   = self.true_label.to(self.device)
        self.false_label  = self.false_label.to(self.device)

    def define_optimizers(self):
        self.optimizerG = optim.Adam(self.G_params, lr=opt.learning_rate, betas=opt.betas)
        self.optimizerD = optim.Adam(self.D_params, lr=opt.learning_rate, betas=opt.betas)

    def backward_G(self):
        self.loss_G = self.criterionGAN(self.result_of_fake, self.true_label)
        self.loss_G.backward()

    def backward_D(self):
        loss_D_real = self.criterionGAN(self.result_of_real, self.true_label)
        loss_D_fake = self.criterionGAN(self.result_of_fake, self.false_label)
        self.loss_D = (loss_D_real + loss_D_fake) / 2
        self.loss_D.backward()

    def forward(self, data: torch.Tensor):
        self.data = data.to(self.device)
        noise = torch.tensor(np.random.normal(size=100))
        self.fake = self.netG(noise)
        self.result_of_fake  = self.netD(self.fake)
        self.result_of_fake_ = self.netD(self.fake.detach())
        self.result_of_real  = self.netD(self.data)

    def backward(self):
        self.optimizerG.zero_grad()
        self.backward_G()
        self.optimizerG.step()

        self.optimizerD.zero_grad()
        self.backward_D()
        self.optimizerD.step()


class Visualizer:

    def __init__(self, model, test_noise: np.ndarray):
        self.model = model
        self.test_noise = test_noise
        self.fig, self.ax = plt.subplots()


if __name__ == '__main__':

    dataset = datasets.MNIST(
        './data/mnist', train=True, download=True,
        transform=transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True)

    model = GAN(lr=opt.learning_rate, betas=opt.betas, n_epochs=opt.n_epochs)

    for epoch in range(opt.n_epochs):
        for batch_idx, (data, _) in enumerate(dataloader):
            model.forward(data)
            model.backward()
