import os
import time
import random
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torchvision.transforms as transforms

from PIL import Image


class Options:
    batch_size = 1
    image_pool_size = 50
    learning_rate = 2e-4
    betas = (.5, .999)
    n_epochs = 200
    ngf = 64  # #filters in the last conv layer
    ndf = 64  # #filters in the last conv layer
    isTrain = True
    # save_dir = './checkpoints'
    lambdaA = 10.
    lambdaB = 10.
    lambdaIdt = .5
    load_size = 286
    begin_decay = 100
    display_freq = 400


opt = Options()


def print_options():
    print(f'\n\n{" OPTIONS ":=^31}')
    for k, v in Options.__dict__.items():
        if not k.startswith('__'):
            print(f'{k:>15}:{v}')
    print(f'{" END ":=^31}\n\n')


def resblock(dim):  # res이기 때문에 앞뒤 채널이 똑같아야 함

    return nn.Sequential(

        nn.ReflectionPad2d(1),
        nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
        nn.BatchNorm2d(dim),
        nn.ReLU(inplace=True),

        nn.ReflectionPad2d(1),
        nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
        nn.BatchNorm2d(dim)

    )


def init_net(net):
    def init_func(m):
        if type(m) is nn.Conv2d:
            nn.init.normal_(m.weight.data, mean=0., std=.02)  # init.normal 은 deprecate 됨
        elif type(m) is nn.BatchNorm2d:
            nn.init.normal_(m.weight.data, mean=1., std=.02)
            nn.init.constant_(m.bias.data, val=0.)

    # net의 각 layer에 recursive하게 적용됨
    # net 자체에도 적용되기 때문에 init_func에서 if로 막아줘야댐
    return net.apply(init_func)  # 이거 return 값 있음?


def define_G(ngf):
    return init_net(Generator(ngf))


def define_D(ndf):
    return init_net(Discriminator(ndf))


class Monet2PhotoDataset(torch.utils.data.Dataset):

    def __init__(self, load_size=286, crop_size=256):
        super().__init__()

        self.transforms = transforms.Compose([
            transforms.Resize([load_size, load_size], Image.BICUBIC),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])

        self.A_dir = os.path.join('monet2photo', 'trainA')
        self.B_dir = os.path.join('monet2photo', 'trainB')
        self.A_paths = os.listdir(self.A_dir)
        self.B_paths = os.listdir(self.B_dir)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

    def __getitem__(self, index) -> tuple:
        """
        :param index: given random integer from DataLoader for data indexing,
                        but I'll ignore in this case.
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

        return A, B  # the data we are going to get while the iteration

    def __len__(self):
        return max(self.A_size, self.B_size)


class ImagePool:
    """
    1. 풀이 비어 있으면 생성된 이미지를 저장 (크기는 batch_size), return 이미지에는 새 이미지 저장
    2. 풀 차있으면 새 이미지마다 1/2 확률로 풀에서 하나 빼서 새 이미지 박음, return에는 구 이미지 저장
    3. 1/2 확률이 안되면 새 이미지 저장
    4. 리턴할 거 텐서로 concat

    This class implements an image buffer that stores previously generated images.

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
            image = torch.unsqueeze(image.data, 0)
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

    def __init__(self, ngf=64):
        super().__init__()

        self.model = nn.Sequential(

            # 크기 안 줄어듦
            nn.ReflectionPad2d(3),
            nn.Conv2d(      3,     ngf, kernel_size=7, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),

            # Downsampling 2회
            nn.Conv2d(    ngf, 2 * ngf, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d( 2 * ngf),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * ngf, 4 * ngf, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d( 2 * ngf),
            nn.ReLU(inplace=True),

            # 9 Res-blocks
            *[resblock(4 * ngf) for _ in range(9)],

            # Upsampling 2회
            nn.ConvTranspose2d(4 * ngf, 2 * ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(          2 * ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2 * ngf,     ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),

            # 크기 안 변함
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0),
            nn.Tanh()

        )

    def forward(self, *input):
        return self.model(*input)


class Discriminator(nn.Module):

    def __init__(self, ndf=64):
        super().__init__()

        self.model = nn.Sequential(

            nn.Conv2d(      3,     ndf, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(.2, True),

            nn.Conv2d(    ndf, 2 * ndf, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(2 * ndf),
            nn.LeakyReLU(.2, True),
            nn.Conv2d(2 * ndf, 4 * ndf, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(2 * ndf),
            nn.LeakyReLU(.2, True),

            nn.Conv2d(4 * ndf, 8 * ndf, kernel_size=4, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(8 * ndf),
            nn.LeakyReLU(.2, True),

            nn.Conv2d(8 * ndf,       1, kernel_size=4, stride=1, padding=1, bias=True)

        )

    def forward(self, *input):
        return self.model(*input)


class CycleGAN:

    def __init__(self, batch_size=1, lr=2e-4, betas=(.5, .999), n_epochs=200,
                 ngf=64, ndf=64, save_dir='./checkpoints',
                 lambdaA=10., lambdaB=10., lambdaIdt=.5):
        self.batch_size = batch_size
        self.learning_rate = lr
        self.betas = betas
        self.n_epochs = n_epochs
        self.ngf = ngf
        self.ndf = ndf
        self.lambdaA = lambdaA  # Cycle Loss의 계수
        self.lambdaB = lambdaB
        self.lambdaIdt = lambdaIdt
        self.save_dir = save_dir

        self.true_label = torch.tensor(1.)
        self.false_label = torch.tensor(0.)

        # need to add data parallel code
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.fakeA_pool = ImagePool(opt.image_pool_size)
        self.fakeB_pool = ImagePool(opt.image_pool_size)

        self.define_nets()
        self.define_criterions()
        self.move_to()
        self.define_optimizers()
        self.define_schedulers()

    def define_nets(self):
        self.netG_A = define_G(self.ngf)
        self.netG_B = define_G(self.ngf)
        self.netD_A = define_D(self.ndf)
        self.netD_B = define_D(self.ndf)

        self.G_params = self.get_G_params()
        self.D_params = self.get_D_params()

    def define_criterions(self):
        self.criterionGAN = nn.MSELoss()  # LSGAN losses
        self.criterionCycle = nn.L1Loss()
        self.criterionIdt = nn.L1Loss()

    def define_optimizers(self):
        self.optimizerG = optim.Adam(self.G_params, lr=opt.learning_rate, betas=opt.betas)
        self.optimizerD = optim.Adam(self.D_params, lr=opt.learning_rate, betas=opt.betas)

    def define_schedulers(self):
        def lambda_rule(epoch):
            return min(1., (epoch - opt.n_epochs) / (opt.begin_decay - opt.n_epochs + 1))
        self.schedulers = [optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
                           for optimizer in [self.optimizerG, self.optimizerD]]

    def get_G_params(self) -> list:
        return list(self.netG_A.parameters()) + list(self.netG_B.parameters())

    def get_D_params(self) -> list:
        return list(self.netD_A.parameters()) + list(self.netD_B.parameters())

    def move_to(self):
        self.netG_A       = self.netG_A.to(self.device)
        self.netG_B       = self.netG_B.to(self.device)
        self.netD_A       = self.netD_A.to(self.device)
        self.netD_B       = self.netD_B.to(self.device)
        self.criterionGAN = self.criterionGAN.to(self.device)
        self.true_label   = self.true_label.to(self.device)
        self.false_label  = self.false_label.to(self.device)

    def backward_G(self):
        self.loss_G_A     = self.criterionGAN(self.netD_A(self.fakeB), self.true_label)
        self.loss_G_B     = self.criterionGAN(self.netD_B(self.fakeA), self.false_label)
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
        pred_real = netD(real)
        pred_fake = netD(fake.detach())
        loss_D_real = self.criterionGAN(pred_real, self.true_label)
        loss_D_fake = self.criterionGAN(pred_fake, self.false_label)
        loss_D = (loss_D_real + loss_D_fake) / 2
        return loss_D

    def compute_loss_D_A(self):
        fake_B = self.fakeB_pool.query(self.fakeB)
        self.loss_D_A = self.compute_loss_D_basic(self.netD_A, self.realB, fake_B)

    def compute_loss_D_B(self):
        fake_A = self.fakeA_pool.query(self.fakeA)
        self.loss_D_B = self.compute_loss_D_basic(self.netD_B, self.realA, fake_A)

    def backward_D(self):
        self.compute_loss_D_A()
        self.compute_loss_D_B()
        self.loss_D = self.loss_D_A + self.loss_D_B
        self.loss_D.backward()

    def forward(self, realA, realB):
        self.realA = realA.to(self.device)
        self.realB = realB.to(self.device)
                                              #   X   <------->   Y
        self.fakeB = self.netG_A(self.realA)  # realA  --G_A--> fakeB
        self.recoA = self.netG_B(self.fakeB)  # recoA <--G_B--  fakeB
        self.fakeA = self.netG_B(self.realB)  # fakeA <--G_B--  realB
        self.recoB = self.netG_A(self.fakeA)  # fakeA  --G_A--> recoB

        # to preserve color composition      #      X              Y
        self.idtA = self.netG_A(self.realB)  # G_B--> idtB   realB ----⌍
        self.idtB = self.netG_B(self.realA)  #  ⌎---- realA   idtA <--G_A

    def backward(self):
        self.optimizerG.zero_grad()
        self.backward_G()
        self.optimizerG.step()

        self.optimizerD.zero_grad()
        self.backward_D()
        self.optimizerD.step()

    def get_current_losses(self):
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
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizerG.param_groups[0]['lr']
        print(f'learning rate = {lr:.7f}')


def print_losses(epoch, iters, t_comp, losses):
    """Print the current losses and the computational time

    :param epoch: current epoch
    :param iters: current training iteration during this epoch
    :param t_comp: computational time per data point
    :param losses: training losses normalized by batch size
    """
    total_iters = epoch * opt.batch_size + iters
    if total_iters % opt.display_freq:
        message = f'(epoch: {epoch:3d}, iters: {iters:3d}, time: {t_comp:.3f})  '
        for name, loss in losses.items():
            message += f'{" |  D" if name is "D" else name}: {loss:.3f} '
        print(message)


if __name__ == '__main__':
    ########################## Monet2Photo Full Implementation ##########################
    dataset = Monet2PhotoDataset()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    model = CycleGAN()

    print_options()
    print('Let\'s begin the training!\n')
    for epoch in range(opt.n_epochs):
        for batch_idx, (dataA, dataB) in enumerate(dataloader):
            t0 = time.time()
            model.forward(dataA, dataB)
            model.backward()
            t1 = time.time()
            print_losses(epoch, batch_idx, (t0 - t1) / len(dataA), model.get_current_losses())
        model.update_learning_rate()

    # todo: add the test code both in and out of the loop
