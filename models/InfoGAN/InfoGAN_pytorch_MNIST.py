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
    result_dir = 'resultsinfoGAN'  # test images will be stored in this path


opt = Options()


class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.FCLayers = nn.Sequential(
            nn.Linear(74, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 7 * 7 * 128, bias=False),
            nn.BatchNorm1d(7 * 7 * 128),
            nn.ReLU(inplace=True)
        )

        self.UpconvLayers = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=True),
            nn.Tanh()
        )

    def forward(self, *input):
        x = self.FCLayers(*input)
        x = x.view(-1, 128, 7, 7)
        x = self.UpconvLayers(x)
        return x


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.ConvLayers = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=True),
            nn.LeakyReLU(.1),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(.1)
        )

        self.FCLayers = nn.Sequential(
            nn.Linear(7 * 7 * 128, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(.1)
        )

        self.D = nn.Linear(1024, 1, bias=True)

        self.Q = nn.Sequential(
            nn.Linear(1024, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(.1),

            nn.Linear(128, 14, bias=True)
        )

        self.D_parameters = [
            *self.ConvLayers.parameters(),
            *self.FCLayers.parameters(),
            *self.D.parameters()
        ]

        self.Q_parameters = self.Q.parameters()

    def forward(self, *input):
        x = self.ConvLayers(*input)
        x = x.view(-1, 7 * 7 * 128)
        x = self.FCLayers(x)

        D_logits = self.D(x)
        Q_logits = self.Q(x)
        Q_logits[:, -2:] = torch.sigmoid(Q_logits[:, -2:])  # to make sure stds to be positive

        return D_logits, Q_logits


def init_net(net) -> nn.Module:
    def init_func(m):
        if type(m) is nn.Conv2d:
            nn.init.normal_(m.weight.data, mean=0., std=.02)  # init.normal has been deprecated
        elif type(m) is nn.BatchNorm2d:
            nn.init.normal_(m.weight.data, mean=1., std=.02)
            nn.init.constant_(m.bias.data, val=0.)

    return net.apply(init_func)  # apply recursively to the net


def define_G():
    return Generator()


def define_D():
    return Discriminator()


class InfoGAN:

    def __init__(
            self,
            lr=2e-4, betas=(.5, .999), n_epochs=50
    ):
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

        self.netG.train()
        self.netD.train()

    def define_criterions(self):
        class GaussianNLLLoss(nn.Module):

            def __init__(self):
                super().__init__()

            def forward(self, c, c_hat, sigma):
                l = (c - c_hat) ** 2
                l /= (2 * sigma ** 2)
                l += torch.log(sigma)
                return l.mean()

        self.criterionGAN = torch.nn.BCEWithLogitsLoss()
        self.criterionCat = torch.nn.CrossEntropyLoss()
        self.criterionCon = GaussianNLLLoss()

    def move_to(self):
        self.netG         = self.netG.to(self.device)
        self.netD         = self.netD.to(self.device)
        self.criterionGAN = self.criterionGAN.to(self.device)
        self.criterionCat = self.criterionCat.to(self.device)
        self.criterionCon = self.criterionCon.to(self.device)

    def define_optimizers(self):
        Q_parameters = [*self.netG.parameters(), *self.netD.D_parameters, *self.netD.Q_parameters]
        self.optimizerD = optim.Adam(self.netD.D_parameters, lr=self.learning_rate, betas=self.betas)
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.learning_rate * 5, betas=self.betas)
        self.optimizerQ = optim.Adam(          Q_parameters, lr=self.learning_rate * 5, betas=self.betas)

    def backward_G(self):
        self.loss_G = self.criterionGAN(self.fake_logits, torch.ones_like(self.fake_logits))
        self.loss_G.backward(retain_graph=True)

    def backward_D(self):
        loss_D_real = self.criterionGAN(self.real_logits,  torch.ones_like(self.real_logits))
        loss_D_fake = self.criterionGAN(self.fake_logits_, torch.zeros_like(self.fake_logits_))
        self.loss_D = (loss_D_real + loss_D_fake) / 2
        self.loss_D.backward(retain_graph=True)

    def backward_Q(self):
        code_disc,   code_cont   = self.random_code[:, :10], self.random_code[:, 10:]  # 10, 2
        disc_logits, cont_logits = self.code_logits[:, :10], self.code_logits[:, 10:]  # 10, 4(2 means, 2 stds)
        loss_Q_disc = self.criterionCat(disc_logits, code_disc.argmax(dim=1))  # target should not be one-hot
        loss_Q_cont = self.criterionCon(code_cont, cont_logits[:, :2], cont_logits[:, 2:])
        self.loss_Q = loss_Q_disc + loss_Q_cont * .15
        self.loss_Q.backward()

    def forward(self, data: torch.Tensor):
        self.data = data.to(self.device)

        self.code_disc = torch.eye(10)[torch.multinomial(torch.ones(10) / 10, len(self.data), replacement=True)].view(-1, 10)  # one-hot
        self.code_cont = (torch.rand(len(self.data), 2) * 2 - 1).view(-1, 2)
        self.random_code = torch.cat((self.code_disc, self.code_cont), dim=1).to(self.device)
        self.noise = torch.randn(len(self.data), 62).to(self.device)
        self.noise = torch.cat((self.noise, self.random_code), dim=1)

        self.fake = self.netG(self.noise)
        self.fake_logits,  self.code_logits = self.netD(self.fake)
        self.fake_logits_, _ = self.netD(self.fake.detach())  # cut gradient flow not to train both G, D at the same time
        self.real_logits,  _ = self.netD(self.data)

    def backward(self):
        self.optimizerG.zero_grad()
        self.backward_G()
        self.optimizerG.step()

        self.optimizerD.zero_grad()
        self.backward_D()
        self.optimizerD.step()

        self.optimizerQ.zero_grad()
        self.backward_Q()
        self.optimizerQ.step()

    def train(self):
        self.netD.train()
        self.netG.train()

    def eval(self):
        self.netD.eval()
        self.netG.eval()

    # =============== For Visualizing ===============
    def test(self, noise: torch.Tensor):
        """For showing images on the plot"""
        self.fake = self.netG(noise.to(self.device))

    def get_current_images(self):
        return self.fake

    def get_current_losses(self) -> dict:
        return {
            'G': self.loss_G,
            'D': self.loss_D,
            'Q': self.loss_Q
        }

    def save(self, PATH):
        torch.save(self.netG, PATH)


class Visualizer:

    def __init__(self, model, test_noise: torch.Tensor):
        self.model = model
        self.test_noise = test_noise
        self.fig, self.ax = plt.subplots(10, 10, figsize=(8, 8))
        plt.ion()
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
        self.model.test(self.test_noise)
        images = self.model.get_current_images()
        self.fig.suptitle(f'epoch {epoch+1}')
        for i in range(10):
            for j in range(10):
                image, = images[10*i+j]
                ax = self.ax[i][j]
                ax.clear()
                ax.set_axis_off()
                # output of tanh is (-1, 1) but needs to be (0, 1) to ax.imshow
                # -1 < y < 1  ==>  0 < y/2 + .5 < 1
                ax.imshow(image.detach().cpu().numpy() / 2 + .5)


if __name__ == '__main__':

    # make MNIST dataset
    dataset = datasets.MNIST(
        './data', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
    # make dataloader that makes you can iterate through the data by the batch size
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True)

    model = InfoGAN()
    print('Finished model initializing')

    test_noise = torch.cat((
        torch.randn(100, 62),
        torch.eye(10)[torch.arange(10).repeat(10)].view(100, 10),  # size(100, 10), ascend through row
        torch.cat([*torch.linspace(-1, 1, 10).repeat(2, 10, 1).transpose(0, 2)], dim=0)  # size(100, 2), ascend through col
    ), dim=1)

    visualizer = Visualizer(model, test_noise)

    visualizer.print_options()
    for epoch in range(opt.n_epochs):
        for batch_idx, (data, _) in enumerate(dataloader):
            model.forward(data)
            model.backward()
            print(f'\rTrain Epoch: {epoch+1:2}/{opt.n_epochs} [{batch_idx * opt.batch_size:5d}/'
                  f'{len(dataset)} '
                  f'{"="* int(100. * batch_idx / len(dataloader) // 2) + ">":50} '
                  f'({100. * batch_idx / len(dataloader):3.0f}%)]  '
                  f'Loss_G: {model.loss_G:.6f} | Loss_D: {model.loss_D:.6f} | '
                  f'Loss_Q: {model.loss_Q:.6f}', end='')
        # visualizer.print_losses(epoch)
        print('')
        model.eval()
        visualizer.print_images(epoch)
        visualizer.save_images(epoch)
        model.train()
    model.save('./infogan.pth')
    plt.show()
