"""
    One-File CycleGANs (monet2photo task)

    This code was created by Hyeo-Geon Lee (Owen Lee)
    to offer beginners of GANs to easier way to absorb the code.

    Please don't hesitate to issue the code
    since every single issue I've got so far was so useful.

    Author:
        name: Haawron - Hyeo-Geon Lee (Owen Lee)
        storage: https://github.com/Haawron

    Referenced:
        paper: https://arxiv.org/abs/1703.10593
        code: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

"""

import os
import time
import random
import argparse

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torchvision.transforms as transforms

from utils.parallel import DataParallelModel, DataParallelCriterion

from PIL import Image


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--image_pool_size', type=int, default=50, help='the size of image buffer which stores previously generated images')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--betas', type=tuple, default=(.5, .999), help='betas of ADAM')
    parser.add_argument('--n_epochs', type=int, default=200, help='total training epochs')
    parser.add_argument('--ngf', type=int, default=64, help='number of filters in the last conv layer of G')
    parser.add_argument('--ndf', type=int, default=64, help='number of filters in the last conv layer of D')
    parser.add_argument('--lambdaA', type=float, default=10., help='coefficient of forward cycle loss')
    parser.add_argument('--lambdaB', type=float, default=10., help='coefficient of backward cycle loss')
    parser.add_argument('--lambdaIdt', type=float, default=.5, help='coefficient of identity loss')
    parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--begin_decay', type=int, default=100, help='number of epoch beginning to decay')
    parser.add_argument('--display_freq', type=int, default=500, help='iteration frequency of showing training results on screen')
    parser.add_argument('--result_dir', type=str, default='resultsCycleGAN', help='directory in which result images will be stored')
    parser.add_argument('--num_worker', type=int, default=4, help='number of workers for Dataloader')
    parser.add_argument('--ckpt_epoch', type=int, default=10, help='saves the model every this epoch')
    parser.add_argument('--resumetrain', type=str, default=None, help='resume training if you pass the .pth file directory path')
    parser.add_argument('--parallel', action='store_true', help='Train the model in parallel way. Not recommended for evaluation')
    parser.add_argument('--saveoff', action='store_true', help='True if you do not want to save the model')
    parser.add_argument('--liveimageoff', action='store_true', help='turn off the live image update with matplotlib')
    parser.add_argument('--useGTK', action='store_true', help='True if you want to run on X11 based background')
    parser.add_argument('--profile', action='store_true', help='Record profiles in txt file')
    return parser.parse_args()


opt = initialize()

if opt.useGTK:
    import matplotlib as mpl
    mpl.use('TKAgg')
import matplotlib.pyplot as plt


def PATH(x):
    return os.path.join(os.path.dirname(__file__), x)


def resblock(dim):
    """Residual block unit of the ResNet

    :param dim: the number of input channels of this block
    """

    return nn.Sequential(

        nn.ReflectionPad2d(1),
        nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
        nn.InstanceNorm2d(dim),
        nn.ReLU(inplace=True),

        nn.ReflectionPad2d(1),
        nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
        nn.InstanceNorm2d(dim)

    )


class Monet2PhotoDataset(torch.utils.data.Dataset):
    """The Dataset for the task converting Monet paintings to photo"""

    def __init__(self, path, load_size=286, crop_size=256):
        """Brings and stores the paths of images

        :param path: ~~~/datasets/monet2photo
        :param load_size: scale images to this size
        :param crop_size: then crop to this size
        """
        super().__init__()

        self.dest = PATH(path)

        if not os.path.exists(self.dest):
            self.__download()

        self.transforms = transforms.Compose([
            transforms.Resize([load_size, load_size], Image.BICUBIC),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])

        self.A_dir = os.path.join(self.dest, 'trainA')
        self.B_dir = os.path.join(self.dest, 'trainB')
        self.A_paths = os.listdir(self.A_dir)
        self.B_paths = os.listdir(self.B_dir)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

    def __getitem__(self, index) -> tuple:
        """Randomly get image set from each domain A, B

        :param index: given random integer from DataLoader for data indexing,
                        but I'll ignore it in this case
        :return: Returns a tuple which contains A, B image tensors
        """

        # indices should be different to avoid fixed pairs.
        A_index = random.randrange(self.A_size)
        B_index = random.randrange(self.B_size)
        A_path = os.path.join(self.A_dir, self.A_paths[A_index])
        B_path = os.path.join(self.B_dir, self.B_paths[B_index])
        A_image = Image.open(A_path).convert('RGB')
        B_image = Image.open(B_path).convert('RGB')
        A = self.transforms(A_image)
        B = self.transforms(B_image)

        return A, B  # the data we are gonna get while the iteration in the main loop

    def __len__(self):
        return max(self.A_size, self.B_size)

    def __download(self):
        import requests as req
        import zipfile

        os.makedirs(self.dest, exist_ok=True)
        # A, B_dir이랑 A, B_path수정하면 되것다.

        print('Processing...')

        url = 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip'
        print('Downloading ' + url)
        zippath = os.path.join(self.dest, 'monet2photo.zip')
        with open(zippath, "wb") as f:
            response = req.get(url)
            f.write(response.content)

        zipped = zipfile.ZipFile(zippath)
        zipped.extractall(os.path.join(self.dest, '..'))
        zipped.close()

        os.remove(zippath)

        print('Done!')


# This class was copied from junyanz's code. I didn't touch anything.
class ImagePool:
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            # image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:  # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:  # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)  # collect all the images and return
        return return_images


class Generator(nn.Module):
    """Generator which uses 9 residual blocks"""

    def __init__(self, ngf=64):
        super().__init__()

        self.model = nn.Sequential(

            # keeps feature size
            nn.ReflectionPad2d(3),
            nn.Conv2d(      3,     ngf, kernel_size=7, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(     ngf),
            nn.ReLU(inplace=True),

            # Downsample twice
            nn.Conv2d(    ngf, 2 * ngf, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d( 2 * ngf),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * ngf, 4 * ngf, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d( 2 * ngf),
            nn.ReLU(inplace=True),

            # 9 Res-blocks
            *[resblock(4 * ngf) for _ in range(9)],

            # Upsample twice
            nn.ConvTranspose2d(4 * ngf, 2 * ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(          2 * ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2 * ngf,     ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(              ngf),
            nn.ReLU(inplace=True),

            # keeps feature size
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0),
            nn.Tanh()

        )

    def forward(self, *input):
        # if isinstance(input, list):
        #     return self.model(torch.cat(input))
        return self.model(*input)


class Discriminator(nn.Module):
    """Discriminator based on 70x70 PatchGAN"""

    def __init__(self, ndf=64):
        super().__init__()

        self.model = nn.Sequential(

            nn.Conv2d(      3,     ndf, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(.2, inplace=True),

            nn.Conv2d(    ndf, 2 * ndf, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d( 2 * ndf),
            nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(2 * ndf, 4 * ndf, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d( 2 * ndf),
            nn.LeakyReLU(.2, inplace=True),

            nn.Conv2d(4 * ndf, 8 * ndf, kernel_size=4, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d( 8 * ndf),
            nn.LeakyReLU(.2, inplace=True),

            nn.Conv2d(8 * ndf,       1, kernel_size=4, stride=1, padding=1, bias=True)

        )

    def forward(self, *input):
        return self.model(*input)


def init_net(net) -> nn.modules:
    """Apply the weight initialization method through the all layers of the net

    :return: initialized net
    """
    def init_func(m):
        if type(m) is nn.Conv2d:
            nn.init.normal_(m.weight.data, mean=0., std=.02)  # init.normal has been deprecated
        # elif type(m) is nn.BatchNorm2d:
        #     nn.init.normal_(m.weight.data, mean=1., std=.02)
        #     nn.init.constant_(m.bias.data, val=0.)

    return net.apply(init_func)  # apply recursively to the net


def define_G(ngf):
    if opt.parallel and torch.cuda.device_count() > 1:
        return DataParallelModel(init_net(Generator(ngf)))
    else:
        return init_net(Generator(ngf))


def define_D(ndf):
    if opt.parallel and torch.cuda.device_count() > 1:
        return DataParallelModel(init_net(Discriminator(ndf)))
    else:
        return init_net(Discriminator(ndf))


class CycleGAN:
    """GANs which use cycle loss to make generators to map the image
     using meaningful connections between domains, which keeps content of the image
     so that generators can train how to do style-transfer without paired dataset.
    """

    def __init__(self, lr=2e-4, betas=(.5, .999), n_epochs=200,
                 ngf=64, ndf=64,
                 lambdaA=10., lambdaB=10., lambdaIdt=.5):
        self.learning_rate = lr
        self.betas = betas
        self.n_epochs = n_epochs
        self.ngf = ngf
        self.ndf = ndf
        self.lambdaA = lambdaA  # coefficients of cycle losses
        self.lambdaB = lambdaB
        self.lambdaIdt = lambdaIdt  # coefficient of Identity losses

        self.true_label = torch.tensor(1.)
        self.false_label = torch.tensor(0.)

        # todo: need to add data parallel code
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.fakeA_pool = ImagePool(opt.image_pool_size)
        self.fakeB_pool = ImagePool(opt.image_pool_size)

        self.define_nets()
        self.define_criterions()
        self.move_to()
        self.define_optimizers()
        self.define_schedulers()

    def define_nets(self):
        """Define generators and discriminators for both directions"""
        self.netG_A = define_G(self.ngf)
        self.netG_B = define_G(self.ngf)
        self.netD_A = define_D(self.ndf)
        self.netD_B = define_D(self.ndf)

        self.G_params = list(self.netG_A.parameters()) + list(self.netG_B.parameters())
        self.D_params = list(self.netD_A.parameters()) + list(self.netD_B.parameters())

    def define_criterions(self):
        """Define criterions of losses
        LSGAN loss for GAN losses, L1 loss for cycle, identity losses

        Identity loss is used only for monet2photo task to keep the color context
        If you had missed this in the paper, refer to section [5.2 - photo generation from paintings]
        """
        self.criterionGAN = nn.MSELoss()  # LSGAN losses
        self.criterionCycle = nn.L1Loss()
        self.criterionIdt = nn.L1Loss()

        if opt.parallel and torch.cuda.device_count() > 1:
            self.criterionGAN = DataParallelCriterion(self.criterionGAN)
            self.criterionCycle = DataParallelCriterion(self.criterionCycle)
            self.criterionIdt = DataParallelCriterion(self.criterionIdt)

    def define_optimizers(self):
        """Define optimizers"""
        self.optimizerG = optim.Adam(self.G_params, lr=self.learning_rate, betas=self.betas)
        self.optimizerD = optim.Adam(self.D_params, lr=self.learning_rate, betas=self.betas)

    def define_schedulers(self):
        """Define schedulers
        for <100 epoch, maintain initial learning rate
        and for >=100 epoch, linearly decay to 0"""
        def lambda_rule(epoch):
            return min(1., (epoch - self.n_epochs) / (opt.begin_decay - self.n_epochs + 1))
        self.schedulers = [optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
                           for optimizer in [self.optimizerG, self.optimizerD]]

    def move_to(self):
        """Move Tensors to cuda if available"""
        self.netG_A       = self.netG_A.to(self.device, non_blocking=True)
        self.netG_B       = self.netG_B.to(self.device, non_blocking=True)
        self.netD_A       = self.netD_A.to(self.device, non_blocking=True)
        self.netD_B       = self.netD_B.to(self.device, non_blocking=True)
        self.criterionGAN = self.criterionGAN.to(self.device, non_blocking=True)
        self.criterionCycle= self.criterionCycle.to(self.device, non_blocking=True)
        self.criterionIdt = self.criterionIdt.to(self.device, non_blocking=True)
        self.true_label   = self.true_label.to(self.device, non_blocking=True)
        self.false_label  = self.false_label.to(self.device, non_blocking=True)

    def backward_G(self):
        """Compute losses and gradients"""
        pred_A = self.netD_A(self.fakeB, parallel=opt.parallel)
        pred_B = self.netD_B(self.fakeA, parallel=opt.parallel)
        self.loss_G_A     = self.criterionGAN(pred_A, self.true_label.repeat(opt.batch_size, *pred_A[0][0].size()))
        self.loss_G_B     = self.criterionGAN(pred_B, self.true_label.repeat(opt.batch_size, *pred_B[0][0].size()))
        self.loss_cycle_A = self.criterionCycle(self.recoA, self.realA)
        self.loss_cycle_B = self.criterionCycle(self.recoB, self.realB)
        self.loss_idt_A   = self.criterionIdt(self.idtA, self.realB)
        self.loss_idt_B   = self.criterionIdt(self.idtB, self.realA)

        self.loss_G = (
              self.loss_G_A
            + self.loss_G_B
            + self.loss_cycle_A * self.lambdaA
            + self.loss_cycle_B * self.lambdaB
            + self.loss_idt_A   * self.lambdaA * self.lambdaIdt
            + self.loss_idt_B   * self.lambdaB * self.lambdaIdt
        )

        self.loss_G.backward()

    def compute_loss_D_basic(self, netD, real, fake):
        """Compute losses of corresponding discriminator"""
        pred_real = netD(real, parallel=opt.parallel)
        pred_fake = netD(fake.detach(), parallel=opt.parallel)
        loss_D_real = self.criterionGAN(pred_real, self.true_label.repeat(opt.batch_size, *pred_real[0][0].size()))
        loss_D_fake = self.criterionGAN(pred_fake, self.false_label.repeat(opt.batch_size, *pred_fake[0][0].size()))
        loss_D = (loss_D_real + loss_D_fake) / 2
        return loss_D

    def compute_loss_D_A(self):
        """Compute the loss of D_A
        Discriminator needs to get an image from the image pool
        """
        fake_B = self.fakeB_pool.query(self.fakeB)
        self.loss_D_A = self.compute_loss_D_basic(self.netD_A, self.realB, fake_B)

    def compute_loss_D_B(self):
        """Compute the loss of D_B
        Discriminator needs to get an image from the image pool
        """
        fake_A = self.fakeA_pool.query(self.fakeA)
        self.loss_D_B = self.compute_loss_D_basic(self.netD_B, self.realA, fake_A)

    def backward_D(self):
        """Compute the final loss of discriminators and gradients"""
        self.compute_loss_D_A()
        self.compute_loss_D_B()
        self.loss_D = self.loss_D_A + self.loss_D_B
        self.loss_D.backward()

    def forward(self, realA: torch.Tensor, realB: torch.Tensor, parallel=opt.parallel):
        """Forward images to the net"""
        self.realA = realA.to(self.device, non_blocking=True)
        self.realB = realB.to(self.device, non_blocking=True)
                                                                 #   X   <------->   Y
        self.fakeB = self.netG_A(self.realA, parallel=parallel)  # realA  --G_A--> fakeB
        self.recoA = self.netG_B(self.fakeB, parallel=parallel)  # recoA <--G_B--  fakeB
        self.fakeA = self.netG_B(self.realB, parallel=parallel)  # fakeA <--G_B--  realB
        self.recoB = self.netG_A(self.fakeA, parallel=parallel)  # fakeA  --G_A--> recoB

        # to preserve color composition                         #      X              Y
        self.idtA = self.netG_A(self.realB, parallel=parallel)  # G_B--> idtB   realB ----⌍
        self.idtB = self.netG_B(self.realA, parallel=parallel)  #  ⌎---- realA   idtA <--G_A

    def backward(self):
        """Optimize the parameters"""
        self.optimizerG.zero_grad()
        self.backward_G()
        self.optimizerG.step()

        self.optimizerD.zero_grad()
        self.backward_D()
        self.optimizerD.step()

    def get_current_images(self) -> dict:
        """Returns lately generated images and input images with names"""
        return {
            'realA': self.realA, 'realB': self.realB,
            'fakeA': self.fakeA, 'fakeB': self.fakeB,
            'recoA': self.recoA, 'recoB': self.recoB,
            'idtA':  self.idtA,  'idtB':  self.idtB
        }

    def get_current_losses(self) -> dict:
        """Returns losses of this step with names"""
        loss_names = ['G', 'G_A', 'G_B',
                      'Cycle_A', 'Cycle_B',
                      'Idt_A', 'Idt_B',
                      'D', 'D_A', 'D_B']
        losses = [self.loss_G, self.loss_G_A, self.loss_G_B,
                  self.loss_cycle_A, self.loss_cycle_B,
                  self.loss_idt_A, self.loss_idt_B,
                  self.loss_D, self.loss_D_A, self.loss_D_B]
        return {loss_name: loss for loss_name, loss in zip(loss_names, losses)}

    def update_learning_rate(self):
        """Update the learning rate at the end of each epoch"""
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizerG.param_groups[0]['lr']
        print(f'learning rate = {lr:.7f}')

    def save(self, epoch, total_time):
        """

        :param epoch: epoch that has just finished. Training will begin from (epoch + 1).
        :param total_time:
        :return:
        """
        savefile = PATH('CycleGAN_ckpt.pth')
        torch.save({
            'epoch': epoch,
            'total time': total_time,
            'G_A': self.netG_A.state_dict(),
            'G_B': self.netG_B.state_dict(),
            'D_A': self.netD_A.state_dict(),
            'D_B': self.netD_B.state_dict(),
            'optimizerG': self.optimizerG.state_dict(),
            'optimizerD': self.optimizerD.state_dict()
        }, savefile)

    def load(self, path='auto') -> tuple:
        checkpoint = torch.load(PATH('CycleGAN_ckpt.pth') if path == 'auto' else path)
        self.netG_A.load_state_dict(checkpoint['G_A'])
        self.netG_B.load_state_dict(checkpoint['G_B'])
        self.netD_A.load_state_dict(checkpoint['D_A'])
        self.netD_B.load_state_dict(checkpoint['D_B'])
        self.optimizerG.load_state_dict(checkpoint['optimizerG'])
        self.optimizerD.load_state_dict(checkpoint['optimizerD'])
        return checkpoint['epoch'], checkpoint['total_time']

    def train(self):
        self.netG_A.train()
        self.netG_B.train()
        self.netD_A.train()
        self.netD_B.train()

    def eval(self):
        self.netG_A.eval()
        self.netG_B.eval()
        self.netD_A.eval()
        self.netD_B.eval()


class Visualizer:
    """Visualizes the current states.

    Methods:
        print_images: shows recently generated images
        print_options: prints options being used in this training.
        print_losses: prints losses of this step
    """

    def __init__(self, model, test_images: tuple):
        """Embed the cycleGAN model and launch matplotlib screen

        :param model: the cycleGAN model
        :param test_images: a single image pair for the test
        """
        self.model = model
        self.test_images = test_images
        self.iteration_time = 0
        self.fig, self.ax = plt.subplots(2, 4, figsize=(20, 10))
        if not opt.liveimageoff:
            plt.pause(1.)

        os.makedirs(PATH(opt.result_dir), exist_ok=True)

    def print_images(self, epoch, iters, batches_per_epoch):
        if not opt.liveimageoff:
            self.__show_images_with_plt(epoch, iters, batches_per_epoch, mode='print')

    def save_images(self, epoch, iters, batches_per_epoch):
        self.__show_images_with_plt(epoch, iters, batches_per_epoch, mode='save')

    def __show_images_with_plt(self, epoch, iters, batches_per_epoch, mode):
        """Shows or saves recently generated images

        :param epoch: current epoch
        :param iters: current iteration count of this epoch
        :param batches_per_epoch: number of batches per epoch
        :param mode: 'save' | 'print'
        """
        assert mode in ['save', 'print']

        total_iters = epoch * batches_per_epoch + iters
        if total_iters % opt.display_freq == 0:
            if mode is 'save':
                self.model.forward(*self.test_images, parallel=False)
            images = list(self.model.get_current_images().items())
            self.fig.suptitle(f'epoch {epoch+1} iter {iters}')
            for i in range(2):
                for j in range(4):
                    name, (image, *_) = images[i+2*j]
                    ax = self.ax[i][j]
                    ax.clear()
                    ax.set_axis_off()
                    ax.imshow(image.detach().cpu().numpy().transpose(1, 2, 0) / 2 + .5)
                    ax.set_title(name)
            if mode is 'save':
                plt.savefig(PATH(f'{opt.result_dir}/{epoch+1:03d}_{iters:04d}.png'), bbox_inches='tight')
            elif mode is 'print':
                plt.pause(.001)

    @staticmethod
    def print_options():
        """Prints options being used in this training."""
        print(f'\n\n{" OPTIONS ":=^41}')
        for k, v in opt.__dict__.items():
            if not k.startswith('__'):  # not for built-in members
                print(f'{k:>20}:{v}')
        print(f'{" END ":=^41}\n\n')

    def print_losses(self, epoch, iters, t_comp, t_global, batches_per_epoch, n_data):
        """Prints the current losses and the computational time

        :param epoch: current epoch
        :param iters: current training iteration during this epoch
        :param t_comp: computational time per batch
        :param t_global: total time spent during this training
        :param batches_per_epoch: number of batches per epoch
        :param n_data: number of total data
        """
        nl = '\n'  # to embed in f-string
        self.iteration_time += t_comp
        total_iters = epoch * batches_per_epoch + iters
        if total_iters % opt.display_freq == 0 and total_iters != 0:
            time_per_data = self.iteration_time / opt.batch_size / opt.display_freq
            eta = (n_data * (opt.n_epochs - epoch) - (iters + 1) * opt.batch_size) * time_per_data
            message = (
                f'[epoch: {epoch+1:3d}, iters: {iters:4d}/{batches_per_epoch}]\n'
                f'{"":12}'
                f'time: {self.iteration_time:.3f}s, time/data: {time_per_data:.3f}s, '
                f'total training time: {self.sec2time(t_global)}, '
                f'ETA: {self.sec2time(eta)}'
            )
            for name, loss in self.model.get_current_losses().items():
                loss_format = '6.3f' if name in ["G", "D"] else '.3f'
                message += (
                    f'{nl + " " * 12 if name in ["G", "D"] else ""}'
                    f'{name}: {loss:{loss_format}} ')
            print(message, '\n')
            self.iteration_time = 0

    @staticmethod
    def sec2time(sec):
        m, s = divmod(int(sec), 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        return (
            f'{f"{d}d " if d != 0 else ""}'
            f'{h}h {m:02d}m {s:02d}s'
        )


def iterate_epoch(model, visualizer, epoch, t0_global, prev_training_time, dataloader, n_data):
    for batch_idx, (dataA, dataB) in enumerate(dataloader):
        if opt.parallel and dataA.size()[0] != opt.batch_size:
            # leftovers(?) like the last minibatch of an epoch may cause an error
            continue
        t0 = time.time()
        model.forward(dataA, dataB)
        model.backward()
        t1 = time.time()
        model.eval()
        visualizer.print_losses(epoch, batch_idx, t1 - t0, t1 - t0_global + prev_training_time, len(dataloader), n_data)
        visualizer.print_images(epoch, batch_idx, len(dataloader))
        visualizer.save_images(epoch, batch_idx, len(dataloader))  # 텐서가 들어가면 scatter 해버려서 뭔가 명시해 줘야댐
        model.train()


########################## Monet2Photo Full Implementation ##########################

def train():
    datapath = os.path.join('..', '..', 'datasets', 'monet2photo')
    dataset = Monet2PhotoDataset(
        datapath, opt.load_size, opt.crop_size)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_worker)

    model = CycleGAN(
        lr=opt.learning_rate, betas=opt.betas, n_epochs=opt.n_epochs,
        ngf=opt.ngf, ndf=opt.ndf,
        lambdaA=opt.lambdaA, lambdaB=opt.lambdaB, lambdaIdt=opt.lambdaIdt
    )

    start_epoch, prev_training_time = model.load(opt.resumetrain) if opt.resumetrain else 0, 0

    get_image = lambda path: dataset.transforms(Image.open(path).convert('RGB')).unsqueeze_(0)
    test_imageA = get_image(os.path.join(PATH(datapath), 'testA', '00050.jpg'))
    test_imageB = get_image(os.path.join(PATH(datapath), 'testB', '2014-12-07 05:00:46.jpg'))
    test_images = test_imageA, test_imageB
    visualizer = Visualizer(model, test_images)

    visualizer.print_options()
    print('Let\'s begin the training!\n')
    t0_global = time.time()
    for epoch in range(start_epoch, opt.n_epochs):
        t0_epoch = time.time()

        if opt.profile:
            with torch.autograd.profiler.profile(use_cuda=True) as prof: iterate_epoch(model, visualizer, epoch, t0_global, prev_training_time, dataloader, len(dataset))
            with open('prof.txt', 'w') as f: f.write(str(prof))
        else: iterate_epoch(model, visualizer, epoch, t0_global, prev_training_time, dataloader, len(dataset))

        if not opt.saveoff and (epoch + 1) % opt.ckpt_epoch == 0:
            model.save(epoch + 1, time.time() - t0_global + prev_training_time)

        print(f'End of Epoch {epoch+1:3d} Time spent: {visualizer.sec2time(time.time()-t0_epoch)}')
        model.update_learning_rate()
        print("=" * 99, '\n\n')
    print(f'End of the Training, Total Time Spent: {visualizer.sec2time(time.time()-t0_global+prev_training_time)}')
    plt.show()

if __name__=='__main__': train()
