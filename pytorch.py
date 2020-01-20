#!/usr/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import datasets

def get_batch(batch_size, image_size, path = 'train/processed_celeba_small/celeba/'):

    transform = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), transforms.ToTensor()])
    dataset = datasets.ImageFolder(path, transform = transform)
    batch = torch.utils.data.DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)
    return batch

batch_size = 256
img_size = 32

trainset = get_batch(batch_size, img_size)

def imshow(img):
    img_np = img.numpy()
    plt.imshow(np.transpose(img_np, (1,2,0)))

dataitem = iter(trainset)
images,nolabel = dataitem.next()

fig = plt.figure(figsize=(10,4))
plot_size = 20
for idx in range(plot_size):
    subplot = fig.add_subplot(2, plot_size/2, idx+1, xticks=[], yticks=[])  #adding
    imshow(images[idx])
#plt.show()     //Displays the plot

def scale(x, feature_range=(-1,1)):
    mn,mx = feature_range
    x = mn+(x-mn)*2/(mx-mn)
    return x

def conv(input_c, output, kernel_size, stride=2, padding=1, batch_norm = True):
    layers=[]
    convl = nn.Conv2d(input_c, output, kernel_size, stride, padding, bias=False)
    layers.append(convl)
    if batch_norm:
        layers.append(nn.BatchNorm2d(output))
    return nn.Sequential(*layers) #unpacking layers

class Discriminator(nn.Module):

    def __init__(self, conv_dim):
        super(Discriminator, self).__init__()   # vs nn.Module.__init__(self) ?
        self.conv_dim = conv_dim
        self.layer_1 = conv(3, conv_dim, 4, batch_norm = False)
        self.layer_2 = conv(conv_dim, conv_dim*2, 4)
        self.layer_3 = conv(conv_dim*2, conv_dim*4, 4)
        self.fc = nn.Linear(conv_dim*4*4*4, 1)

    def forward(self, x):
        x = F.leaky_relu(self.layer_1(x))
        x = F.leaky_relu(self.layer_2(x))
        x = F.leaky_relu(self.layer_3(x))
        x = x.view(-1, self.conv_dim*4*4*4)
        x = self.fc(x)
        return x


def deconv(input_c, output, kernel_size, stride=2, padding=1, batch_norm = True):
    layers = []
    decon = nn.ConvTranspose2d(input_c, output, kernel_size, stride, padding, bias=False)
    layers.append(decon)
    if batch_norm:
        layers.append(nn.BatchNorm2d(output))
    return nn.Sequential(*layers)

class Generator(nn.Module):
    def __init__(self, z_size, conv_dim):
        super(Generator, self).__init__()
        self.conv_dim = conv_dim
        self.fc = nn.Linear(z_size, conv_dim*8*2*2)
        self.layer_1 = deconv(conv_dim*8, conv_dim*4, 4)
        self.layer_2 = deconv(conv_dim*4, conv_dim*2, 4)
        self.layer_3 = deconv(conv_dim*2, conv_dim, 4)
        self.layer_4 = deconv(conv_dim, 3, 4, batch_norm = False)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.conv_dim*8, 2, 2)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = torch.tanh(self.layer_4(x))
        return x

def weights_init_normal(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != 1 or classname.find('Linear') != -1):
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.zero_()

def build_network(d_conv_dim, g_conv_dim, z_size):
    D = Discriminator(d_conv_dim)
    G = Generator(z_size=z_size, conv_dim=g_conv_dim)
    
    D.apply(weights_init_normal)
    G.apply(weights_init_normal)
    print(D)
    print()
    print(G)
    return D, G

d_conv_dim = 64
g_conv_dim = 64
z_size = 100

D, G = build_network(d_conv_dim, g_conv_dim, z_size)

def real_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.ones(batch_size)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss

d_optimizer = torch.optim.Adam(D.parameters(), lr = .0002, betas = [0.5, 0.999])
g_optimizer = torch.optim.Adam(G.parameters(), lr = .0002, betas = [0.5, 0.999])


def train(D, G, n_epochs, print_every=50):
    samples = []
    losses = []
    sample_size = 16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()

    for epoch in range(n_epochs):
        print"epoch: ", epoch
        for batch_i, (real_images, _) in enumerate(trainset):
            batch_size = real_images.size(0)
            real_images = scale(real_images)
            
            d_optimizer.zero_grad()
            d_out_real = D(real_images)
            z = np.random.uniform(-1, 1, size = (batch_size, z_size))
            z = torch.from_numpy(z).float()
            d_loss = real_loss(d_out_real) + fake_loss(D(G(z)))
            d_loss.backward()
            d_optimizer.step()

            G.train()
            g_optimizer.zero_grad()
            z = np.random.uniform(-1, 1, size = (batch_size, z_size))
            z = torch.from_numpy(z).float()
            g_loss = real_loss(D(G(z)))
            g_loss.backward()
            g_optimizer.step()
            
            if batch_i % print_every==0:
                losses.append((d_loss.item(), g_loss.item()))
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(epoch+1, n_epochs, d_loss.item(), g_loss.item()))            
    G.eval()
    samples_z = G(fixed_z)
    samples.append(samples_z)
    G.train()

    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)

    return losses

n_epochs = 40
losses = train(D, G, n_epochs=n_epochs)
