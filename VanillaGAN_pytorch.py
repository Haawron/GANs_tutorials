import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


class Options:
    batch_size = 256
    learning_rate = 2e-3
    betas = (.5, .999)
    n_epochs = 100
    result_dir = 'resultsGAN'


opt = Options()


class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, *input):
        return self.model(*input)


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, *input):
        return self.model(*input)


def init_net(net) -> nn.modules:
    """Apply the weight initialization method through the all layers of the net

    :return: initialized net
    """
    def init_func(m):
        if type(m) is nn.Linear:
            nn.init.normal_(m.weight.data, mean=0., std=.01)  # init.normal has been deprecated
            nn.init.normal_(m.bias.data, mean=0., std=.01)

    return net.apply(init_func)  # apply recursively to the net


def define_G():
    return init_net(Generator())


def define_D():
    return init_net(Discriminator())


class GAN:

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

        self.G_params = list(self.netG.parameters())
        self.D_params = list(self.netD.parameters())

    def define_criterions(self):
        self.criterionGAN = torch.nn.BCELoss()

    def move_to(self):
        self.netG         = self.netG.to(self.device)
        self.netD         = self.netD.to(self.device)
        self.criterionGAN = self.criterionGAN.to(self.device)

    def define_optimizers(self):
        self.optimizerG = optim.Adam(self.G_params, lr=opt.learning_rate, betas=opt.betas)
        self.optimizerD = optim.Adam(self.D_params, lr=opt.learning_rate, betas=opt.betas)

    def backward_G(self):
        self.loss_G = self.criterionGAN(self.result_of_fake, self.true_label)
        self.loss_G.backward()

    def backward_D(self):
        loss_D_real = self.criterionGAN(self.result_of_real, self.true_label)
        loss_D_fake = self.criterionGAN(self.result_of_fake_, self.false_label)
        self.loss_D = (loss_D_real + loss_D_fake) / 2
        self.loss_D.backward()

    def forward(self, data: torch.Tensor):
        self.data = data.view(-1, 784).to(self.device)
        noise = torch.randn(len(self.data), 128).to(self.device)
        self.fake = self.netG(noise)
        self.result_of_fake  = self.netD(self.fake)
        self.result_of_fake_ = self.netD(self.fake.detach())
        self.result_of_real  = self.netD(self.data)
        self.true_label = torch.ones_like(self.result_of_real).to(self.device)
        self.false_label = torch.zeros_like(self.result_of_fake).to(self.device)

    def eval(self, noise: torch.Tensor):
        self.fake = self.netG(noise.to(self.device))

    def backward(self):
        self.optimizerG.zero_grad()
        self.backward_G()
        self.optimizerG.step()

        self.optimizerD.zero_grad()
        self.backward_D()
        self.optimizerD.step()

    def get_current_images(self):
        return self.fake

    def get_current_losses(self) -> dict:
        return {'G': self.loss_G, 'D': self.loss_D}


class Visualizer:

    def __init__(self, model, test_noise: torch.Tensor):
        self.model = model
        self.test_noise = test_noise
        self.fig, self.ax = plt.subplots(5, 5, figsize=(8, 8))
        plt.pause(.001)

        if opt.result_dir not in os.listdir():
            os.mkdir(opt.result_dir)

    @staticmethod
    def print_options():
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
        self.__show_images_with_plt(epoch, mode='print')

    def save_images(self, epoch):
        self.__show_images_with_plt(epoch, mode='save')

    def __show_images_with_plt(self, epoch, mode):
        assert mode in ['save', 'print']

        if mode is 'save':  # I love "is" more than "==" since more readable
            self.model.eval(self.test_noise)
        images = self.model.get_current_images()
        self.fig.suptitle(f'epoch {epoch}')
        for i in range(5):
            for j in range(5):
                image = images[5*i+j].view(28, 28)
                ax = self.ax[i][j]
                ax.clear()
                ax.set_axis_off()
                ax.imshow(image.detach().cpu().numpy() / 2 + .5)
        if mode is 'save':
            plt.savefig(f'{opt.result_dir}/{epoch:02d}.png', bbox_inches='tight')
        elif mode is 'print':
            plt.pause(.001)


if __name__ == '__main__':

    dataset = datasets.MNIST(
        './data/mnist', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.5,), (.5,))
        ]))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True)

    model = GAN(lr=opt.learning_rate, betas=opt.betas, n_epochs=opt.n_epochs)

    test_noise = torch.randn(25, 128)
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
