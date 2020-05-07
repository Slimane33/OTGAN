
from time import time
from multiprocessing import cpu_count
from IPython import display
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image, resize, to_tensor
from torchvision.transforms.functional import normalize
import imageio


class DoubleBatchDataset(data.Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return self.dataset1.shape[0]

    def __getitem__(self, index):
        return self.dataset1[index], self.dataset2[index]


def load_mnist(batch_size=128, img_size=28, double_batch=False):
    """Download, preprocess and load MNIST data."""
    mnist = datasets.MNIST('data', train=True, download=True).data
    # Perform transformation directly on raw data rather than in the DataLoader
    # => avoids overhead of performing transforms at each batch call
    # => much faster epochs.
    pics = []
    for pic in mnist:
        pic = to_pil_image(pic)
        if img_size != 28:
            pic = resize(pic, img_size) # Resize image if needed
        pic = to_tensor(pic) # Tensor conversion normalizes in [0,1]
        pic = normalize(pic, [0.5], [0.5]) # Normalize values in [-1,1]
        pics.append(pic)

    mnist = torch.stack(pics)

    if double_batch:
        mnist = DoubleBatchDataset(mnist, mnist)

    return torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True)


class GANGenerator(nn.Module):
    def __init__(self, input_size, d, output_shape):
        super(GANGenerator, self).__init__()

        self.map1 = nn.Linear(input_size, d)
        self.map2 = nn.Linear(self.map1.out_features, d * 2)
        self.map3 = nn.Linear(self.map2.out_features, d * 4)
        self.map4 = nn.Linear(self.map3.out_features,
                              output_shape[0] * output_shape[1] * output_shape[2])

        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.output_shape = output_shape

    def forward(self, x):
        x = self.act(self.map1(x))
        x = self.act(self.map2(x))
        x = self.act(self.map3(x))
        x = torch.tanh(self.map4(x))

        return x.view((-1,) + self.output_shape)


class GANCritic(nn.Module):
    def __init__(self, input_size, d):
        super(GANCritic, self).__init__()

        self.map1 = nn.Linear(input_size, d)
        self.map2 = nn.Linear(self.map1.out_features, d//2)
        self.map3 = nn.Linear(self.map2.out_features, d//4)
        self.map4 = nn.Linear(self.map3.out_features, 32)

        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = nn.Flatten()(x)
        x = self.act(self.map1(x))
        x = F.dropout(x, 0.3)
        x = self.act(self.map2(x))
        x = F.dropout(x, 0.3)
        x = self.act(self.map3(x))
        x = F.dropout(x, 0.3)
        x = self.map4(x)

        return x


class OTGAN():

    def __init__(self, dataloader, generator, critic, lr=0.0001):

        self.dataloader = dataloader

        # default parameters for mnist
        self.img_channels = dataloader.dataset[0][0].shape[0]
        self.img_rows = dataloader.dataset[0][0].shape[1]
        self.img_cols = dataloader.dataset[0][0].shape[2]
        self.img_shape = (self.img_channels, self.img_rows, self.img_cols)
        self.z_dim = z_dim
        self.lr = lr

        self.generator = generator.to(device)
        self.critic = critic.to(device)

    def sample_data(n_sample):
        z_random = np.random.randn(n_sample, z_dim)
        z_random = torch.FloatTensor(z_random)
        samples = generator(z_random)
        samples = samples.detach().cpu().numpy()
        return samples

    def cost( batch_1, batch_2):
        norm_1 = torch.norm(batch_1, p=2, dim=1).reshape(-1, 1)
        norm_2 = torch.norm(batch_2, p=2, dim=1).reshape(-1, 1)
        return 1 - (torch.matmul(batch_1, batch_2.transpose(0, 1)) /
                  (torch.matmul(norm_1, norm_2.transpose(0, 1))))

    def sinkhorn( a, b, C, reg=0.001, max_iters=100):

        K = torch.exp(-C / reg)
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        for i in range(max_iters):
            u = a / torch.matmul(K, v)
            v = b / torch.matmul(K.T, u)
        return torch.matmul(torch.diag_embed(u),
                            torch.matmul(K, torch.diag_embed(v)))

    def train(self, epochs=100, print_interval=10, save_generator_path=None):

        criterion = nn.BCELoss()  # http://pytorch.org/docs/nn.html#bceloss
        c_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr,
                                 betas=(0.5, 0.999))
        g_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr,
                                 betas=(0.5, 0.999))
        d_steps = 1
        g_steps = 1

        for epoch in range(epochs):

            t = time()
            loss_to_display = []

            for real_1, real_2 in self.dataloader:
                batch_size = real_1.shape[0]

                self.critic.zero_grad()
                self.generator.zero_grad()

                real_1 = real_1.type(torch.FloatTensor)
                real_2 = real_2.type(torch.FloatTensor)

                z1 = torch.FloatTensor(np.random.randn(batch_size,
                                                       self.z_dim))
                fake_1 = self.generator(z1)
                z2 = torch.FloatTensor(np.random.randn(batch_size,
                                                       self.z_dim))
                fake_2 = self.generator(z2)

                print(real_1)
                critic_real_1 = self.critic(real_1)
                critic_real_2 = self.critic(real_2)
                critic_fake_1 = self.critic(fake_1)
                critic_fake_2 = self.critic(fake_2)

                # Computing all matrices of costs

                costs = torch.zeros((4, 4, batch_size, batch_size)).to(device)

                costs[0, 1] = self.cost(critic_real_1, critic_real_2)
                costs[0, 2] = self.cost(critic_real_1, critic_fake_1)
                costs[0, 3] = self.cost(critic_real_1, critic_fake_2)
                costs[1, 2] = self.cost(critic_real_2, critic_fake_1)
                costs[1, 3] = self.cost(critic_real_2, critic_fake_2)
                costs[2, 3] = self.cost(critic_fake_1, critic_fake_2)

                # Computing optimal plans for all costs

                a = (torch.ones(batch_size) / batch_size).to(device)
                b = (torch.ones(batch_size) / batch_size).to(device)

                plans = torch.zeros((4, 4, batch_size, batch_size)).to(device)

                plans[0, 1] = self.sinkhorn(a, b, costs[0, 1], reg=0.01)
                plans[0, 2] = self.sinkhorn(a, b, costs[0, 2], reg=0.01)
                plans[0, 3] = self.sinkhorn(a, b, costs[0, 3], reg=0.01)
                plans[1, 2] = self.sinkhorn(a, b, costs[1, 2], reg=0.01)
                plans[1, 3] = self.sinkhorn(a, b, costs[1, 3], reg=0.01)
                plans[2, 3] = self.sinkhorn(a, b, costs[2, 3], reg=0.01)

                # Computing losses

                losses = torch.zeros((4, 4)).to(device)

                losses[0, 1] = torch.sum(plans[0, 1] * costs[0, 1])
                losses[0, 2] = torch.sum(plans[0, 2] * costs[0, 2])
                losses[0, 3] = torch.sum(plans[0, 3] * costs[0, 3])
                losses[1, 2] = torch.sum(plans[1, 2] * costs[1, 2])
                losses[1, 3] = torch.sum(plans[1, 3] * costs[1, 3])
                losses[2, 3] = torch.sum(plans[2, 3] * costs[2, 3])

                loss = losses[0, 2] + losses[0, 3] + losses[1, 2] + losses[1, 3] - 2 * losses[0, 1] - 2 * losses[2, 3]

                loss.backward()
                c_optimizer.step()
                g_optimizer.step()

                loss_to_display.append(float(loss.detach().cpu().numpy()))

            if (epoch > 0 and epoch % print_interval == 0) or epoch + 1 == epochs:
                de = d_train_loss.detach().cpu().numpy()
                ge = g_error.detach().cpu().numpy()
                print("Epoch %s: C_loss =  %s ;  G_loss = %s;  time = %s" %
                      (epoch, de, ge, time() - t))

            if epoch % 1 == 0:
                samples = self.sample_data(3) * 0.5 + 0.5
                for img in samples:
                    plt.figure()
                    plt.imshow(img[0, :, :], cmap='gray')
                    plt.show()

        if save_generator_path is not None:
            torch.save(self.generator.state_dict(), save_generator_path)



# Vanilla GAN parameters
img_size = 28
z_dim = 32
G_dim_init = 128
C_dim_init = 1024

lr = 0.0002
batch_size = 128
n_epochs = 100

# Get MNIST data as Torch dataloader
mnist_dataloader = load_mnist(batch_size=batch_size, img_size=img_size,
                              double_batch=True)
img_shape = mnist_dataloader.dataset[0][0].shape
n_pixels = img_shape[0] * img_shape[1] * img_shape[2]

dataloader = mnist_dataloader
img_channels = dataloader.dataset[0][0].shape[0]
img_rows = dataloader.dataset[0][0].shape[1]
img_cols = dataloader.dataset[0][0].shape[2]
img_shape = (img_channels, img_rows, img_cols)
z_dim = z_dim
lr = lr

generator = GANGenerator(z_dim, G_dim_init, img_shape)
critic = GANCritic(n_pixels, C_dim_init)

c_optimizer = optim.Adam(critic.parameters(), lr=lr,
                                 betas=(0.5, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=lr,
                         betas=(0.5, 0.999))
real_1, real_2 = next(iter(dataloader))

batch_size = real_1.shape[0]

critic.zero_grad()
generator.zero_grad()

real_1 = real_1.type(torch.FloatTensor)
real_2 = real_2.type(torch.FloatTensor)

z1 = torch.FloatTensor(np.random.randn(batch_size,
                                       z_dim))
fake_1 = generator(z1)
z2 = torch.FloatTensor(np.random.randn(batch_size,
                                       z_dim))
fake_2 = generator(z2)

critic_real_1 = critic(real_1)
critic_real_2 = critic(real_2)
critic_fake_1 = critic(fake_1)
critic_fake_2 = critic(fake_2)

# Computing all matrices of costs

costs = torch.zeros((4, 4, batch_size, batch_size))

costs[0, 1] = cost(critic_real_1, critic_real_2)
costs[0, 2] = cost(critic_real_1, critic_fake_1)
costs[0, 3] = cost(critic_real_1, critic_fake_2)
costs[1, 2] = cost(critic_real_2, critic_fake_1)
costs[1, 3] = cost(critic_real_2, critic_fake_2)
costs[2, 3] = cost(critic_fake_1, critic_fake_2)

# Computing optimal plans for all costs

a = (torch.ones(batch_size) / batch_size)
b = (torch.ones(batch_size) / batch_size)

plans = torch.zeros((4, 4, batch_size, batch_size))

plans[0, 1] = sinkhorn(a, b, costs[0, 1], reg=0.01)
plans[0, 2] = sinkhorn(a, b, costs[0, 2], reg=0.01)
plans[0, 3] = sinkhorn(a, b, costs[0, 3], reg=0.01)
plans[1, 2] = sinkhorn(a, b, costs[1, 2], reg=0.01)
plans[1, 3] = sinkhorn(a, b, costs[1, 3], reg=0.01)
plans[2, 3] = sinkhorn(a, b, costs[2, 3], reg=0.01)

# Computing losses

losses = torch.zeros((4, 4))

losses[0, 1] = torch.sum(plans[0, 1] * costs[0, 1])
losses[0, 2] = torch.sum(plans[0, 2] * costs[0, 2])
losses[0, 3] = torch.sum(plans[0, 3] * costs[0, 3])
losses[1, 2] = torch.sum(plans[1, 2] * costs[1, 2])
losses[1, 3] = torch.sum(plans[1, 3] * costs[1, 3])
losses[2, 3] = torch.sum(plans[2, 3] * costs[2, 3])

loss = losses[0, 2] + losses[0, 3] + losses[1, 2] + losses[1, 3] - 2 * losses[0, 1] - 2 * losses[2, 3]

loss.backward()
c_optimizer.step()
g_optimizer.step()