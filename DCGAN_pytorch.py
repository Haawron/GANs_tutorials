import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


class Options:
    batch_size = 128  # number of data in one mini-batch
    learning_rate = 2e-4
    betas = (.5, .999)  # betas for Adam Optimizer
    n_epochs = 50  # total number of training epochs
    ngf = 128  # #filters in the last conv layer
    ndf = 128  # #filters in the last conv layer

    image_size = 64
    result_dir = 'resultsDCGAN'  # test images will be stored in this path


opt = Options()


class Generator(nn.Module):

    def __init__(self, ngf=opt.ngf):
        super().__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(    128, 8 * ngf, 4, 1, 0, bias=False),
            nn.BatchNorm2d(             8 * ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8 * ngf, 4 * ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(             4 * ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(4 * ngf, 2 * ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(             2 * ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2 * ngf,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(                 ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(    ngf,       1, 4, 2, 1, bias=True),
            nn.Tanh()
        )

    def forward(self, *input):
        return self.model(*input)


class Discriminator(nn.Module):

    def __init__(self, ndf=opt.ndf):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(      1,     ndf, 4, 2, 1, bias=True),
            nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(    ndf, 2 * ndf, 4, 2, 2, bias=False),
            nn.BatchNorm2d(    2 * ndf),
            nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(2 * ndf, 4 * ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(    4 * ndf),
            nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(4 * ndf, 8 * ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(    8 * ndf),
            nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(8 * ndf,       1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, *input):
        return self.model(*input).squeeze()


def init_net(net) -> nn.Module:
    def init_func(m):
        if type(m) is nn.Conv2d:
            nn.init.normal_(m.weight.data, mean=0., std=.02)  # init.normal has been deprecated
        elif type(m) is nn.BatchNorm2d:
            nn.init.normal_(m.weight.data, mean=1., std=.02)
            nn.init.constant_(m.bias.data, val=0.)

    return net.apply(init_func)  # apply recursively to the net


def define_G():
    return init_net(Generator())


def define_D():
    return init_net(Discriminator())


class DCGAN:

    def __init__(self,
                 lr=2e-4, betas=(.5, .999), n_epochs=50,
                 ngf=opt.ngf, ndf=opt.ndf):
        self.learning_rate = lr
        self.betas = betas
        self.n_epochs = n_epochs
        self.ngf = ngf
        self.ndf = ndf

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.define_nets()
        self.define_criterions()
        self.move_to()
        self.define_optimizers()

    def define_nets(self):
        self.netG = define_G()
        self.netD = define_D()

    def define_criterions(self):
        self.criterionGAN = torch.nn.BCELoss()

    def move_to(self):
        self.netG         = self.netG.to(self.device)
        self.netD         = self.netD.to(self.device)
        self.criterionGAN = self.criterionGAN.to(self.device)

    def define_optimizers(self):
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.learning_rate, betas=self.betas)
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.learning_rate, betas=self.betas)

    def backward_G(self):
        true_label = torch.ones_like(self.result_of_fake).to(self.device)
        self.loss_G = self.criterionGAN(self.result_of_fake, true_label)
        self.loss_G.backward()

    def backward_D(self):
        true_label = torch.ones_like(self.result_of_real).to(self.device)
        false_label = torch.ones_like(self.result_of_fake_).to(self.device)
        loss_D_real = self.criterionGAN(self.result_of_real, true_label)
        loss_D_fake = self.criterionGAN(self.result_of_fake_, false_label)
        self.loss_D = (loss_D_real + loss_D_fake) / 2
        self.loss_D.backward()

    def forward(self, data: torch.Tensor):
        self.data = data.to(self.device)
        noise = torch.randn(len(self.data), 128, 1, 1).to(self.device)
        self.fake = self.netG(noise)
        self.result_of_fake  = self.netD(self.fake)
        self.result_of_fake_ = self.netD(self.fake.detach())  # cut gradient flow not to train both G, D at once
        self.result_of_real  = self.netD(self.data)

    def backward(self):
        self.optimizerG.zero_grad()
        self.backward_G()
        self.optimizerG.step()

        self.optimizerD.zero_grad()
        self.backward_D()
        self.optimizerD.step()

    # These methods are for visualizing
    def eval(self, noise: torch.Tensor):
        """For showing images on the plot"""
        self.fake = self.netG(noise.to(self.device))
    def get_current_images(self):
        return self.fake
    def get_current_losses(self) -> dict:
        return {'G': self.loss_G, 'D': self.loss_D}


class Visualizer:

    def __init__(self, model, test_noise: torch.Tensor):
        self.model = model
        self.test_noise = test_noise
        self.fig, self.ax = plt.subplots(5, 5, figsize=(8, 8))
        plt.pause(.001)  # you can think it as updating

        # make result directory if you don't have it
        os.makedirs(opt.result_dir, exist_ok=True)

    @staticmethod
    def print_options():
        """It would be useful to print used options on the console
        when doing an experiment."""
        print(f'\n\n{" OPTIONS ":=^31}')
        for k, v in Options.__dict__.items():
            if not k.startswith('__'):  # not for built-in members
                print(f'{k:>15}:{v}')
        print(f'{" END ":=^31}\n\n')

    def print_losses(self, epoch):
        message = f'epoch {epoch:2d}'
        for name, loss in self.model.get_current_losses().items():
            message += f'  |  {name}: {loss:6.3f}'
        print(message)

    def print_images(self, epoch):
        self.__show_images_with_plt(epoch)
        plt.pause(.001)

    def save_images(self, epoch):
        self.__show_images_with_plt(epoch)
        plt.savefig(f'{opt.result_dir}/{epoch:02d}.png', bbox_inches='tight')

    def __show_images_with_plt(self, epoch):
        self.model.eval(self.test_noise)
        images = self.model.get_current_images()
        self.fig.suptitle(f'epoch {epoch}')
        for i in range(5):
            for j in range(5):
                image, = images[5*i+j]
                ax = self.ax[i][j]
                ax.clear()
                ax.set_axis_off()
                # output of tanh is (-1, 1) but needs to be (0, 1) to ax.imshow
                # -1 < y < 1  ==>  0 < y/2 + .5 < 1
                ax.imshow(image.detach().cpu().numpy() / 2 + .5)


if __name__ == '__main__':

    # make MNIST dataset
    dataset = datasets.MNIST(
        './data/mnist', train=True, download=True,
        transform=transforms.Compose([
            transforms.Resize(opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize((.5,), (.5,))
        ]))
    # make dataloader that makes you can iterate through the data by the batch size
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True)

    model = DCGAN(lr=opt.learning_rate, betas=opt.betas, n_epochs=opt.n_epochs,
                  ngf=opt.ngf, ndf=opt.ndf)

    test_noise = torch.randn(25, 128, 1, 1)  # 5 x 5 latent vectors for test
    visualizer = Visualizer(model, test_noise)

    visualizer.print_options()
    for epoch in range(opt.n_epochs):
        for batch_idx, (data, _) in enumerate(dataloader):
            model.forward(data)
            model.backward()
            visualizer.print_losses(epoch)
            visualizer.print_images(epoch)
            visualizer.save_images(epoch)
    plt.show()
