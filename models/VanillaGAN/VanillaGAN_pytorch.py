"""
    One-File Vanilla GANs

    This code was created by Hyeo-Geon Lee (Owen Lee)
    to offer beginners of GANs to easier way to absorb the code.

    Please don't hesitate to issue the code
    since every single issue I've got so far was so useful.

    Author:
        name: Haawron - Hyeo-Geon Lee (Owen Lee)
        storage: https://github.com/Haawron

    Referenced:
        paper: https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf
"""

import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


class Options:
    """You will have many chances to see author of paper's full implementation codes
    using 'argparse' package. For those who are not friendly with that package, I implemented
    argparse-like but easy to read with this class.
    """
    batch_size = 256  # number of data in one mini-batch
    learning_rate = 2e-3
    betas = (.5, .999)  # betas for Adam Optimizer
    n_epochs = 100  # total number of training epochs
    result_dir = 'resultsGAN'  # test images will be stored in this path


opt = Options()  # You may think it as "opt = args.parse_args()"


class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, *input):
        return self.model(*input)


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, *input):
        return self.model(*input)


def init_net(net) -> nn.Module:
    """Apply the weight initialization method through the all layers of the net

    INITIALIZING WELL IS SUPER IMPORTANT that you can use small nets,
    which have the same performance with big ones.

    PyTorch's default initializer is Xavier or He which is so much useful,
    but that is not the case here. If you want to see how it is important,
    delete this function.

    :return: initialized net
    """
    def init_func(m):
        if type(m) is nn.Linear:
            nn.init.normal_(m.weight.data, mean=0., std=.01)  # init.normal has been deprecated
            nn.init.constant_(m.bias.data, val=0.)

    return net.apply(init_func)  # apply recursively to the net


def define_G():
    return init_net(Generator())


def define_D():
    return init_net(Discriminator())


class GAN:
    """The main GAN model

    You can implement in procedure-oriented way but
    it has many advantages if done in OOP.

    Get friendly with OOP deep learning codes so that
    you can be easy to read HUGE codes of famous authors.
    """

    def __init__(self, lr=2e-4, betas=(.5, .999), n_epochs=100):
        self.learning_rate = lr
        self.betas = betas
        self.n_epochs = n_epochs

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
        self.data = data.view(-1, 784).to(self.device)
        noise = torch.randn(len(self.data), 128).to(self.device)
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
    """Shows Options, current losses and test images

    It would be kind of a mess if you implement these
    in the main scope without this class.
    """

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
                image = images[5*i+j].view(28, 28)
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
            transforms.ToTensor(),
            transforms.Normalize((.5,), (.5,))
        ]))
    # make dataloader that makes you can iterate through the data by the batch size
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True)

    model = GAN(lr=opt.learning_rate, betas=opt.betas, n_epochs=opt.n_epochs)

    test_noise = torch.randn(25, 128)  # 5 x 5 latent vectors for test
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
