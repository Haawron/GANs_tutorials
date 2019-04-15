from .InfoGAN_pytorch_MNIST import Generator
import os
import torch
import matplotlib.pyplot as plt


result_path = 'save_fig'
os.makedirs(result_path, exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.load('infogan.pth')
model.eval()
model.to(device)

fig, axes = plt.subplots(2, 10, figsize=(8, 3))
image_noise = torch.randn(10, 62)
for k in range(100):

    test_noise_cont1 = torch.cat((
        image_noise,
        torch.eye(10)[torch.arange(10)],
        torch.ones(10, 1) * torch.linspace(-2, 2, 100)[k],
        -torch.ones(10, 1)
    ), dim=1).to(device)

    test_noise_cont2 = torch.cat((
        image_noise,
        torch.eye(10)[torch.arange(10)],
        -torch.ones(10, 1),
        torch.ones(10, 1) * torch.linspace(-2, 2, 100)[k]
    ), dim=1).to(device)

    images = [model(test_noise_cont1), model(test_noise_cont2)]
    for i in range(2):
        for j in range(10):
            image, = images[i][j]
            ax = axes[i][j]
            ax.clear()
            ax.set_axis_off()
            # output of tanh is (-1, 1) but needs to be (0, 1) to ax.imshow
            # -1 < y < 1  ==>  0 < y/2 + .5 < 1
            ax.imshow(image.detach().cpu().numpy() / 2 + .5)
    plt.savefig(f'{result_path}/{k}.png', bbox_inches='tight')
    print(f'finished {k+1:3d}{["st", "nd", "rd", "th"][min(3, k % 10)]} image')

from moviepy.editor import *
clip = ImageSequenceClip([f'{result_path}/{x}.png' for x in range(100)], fps=15)
clip.write_gif(f'{result_path}/infogan.gif')
