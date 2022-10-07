import os
import torch
from torchvision.utils import save_image, make_grid
import torchvision.transforms as T
from torchvision.datasets import MNIST, ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt

def main():

    # Root directory for dataset
    dataroot = "dataset/WikiArtDataset/Impressionism2"

    # Number of workers for dataloader
    workers = 4

    # Batch size during training
    batch_size = 128

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 64

    # Size of z latent vector (i.e. size of generator input)
    nz = 64

    # Create the dataset
    dataset = ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)

    # Helper functions
    def get_default_device():
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) # to normalize images valus between -1,1 instead of 0,1
    def denorm(image_tensor):
        return image_tensor * stats[1][0] + stats[0][0]

    def to_device(data, device):
        if isinstance(data, (list, tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

    class DeviceDataLoader():
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device
        
        def __iter__(self):
            for b in self.dl:
                yield to_device(b, self.device)

        def __len__(self):
            return len(self.dl)
    
    
    device = get_default_device()
    print('Device:', device)
    dataloader = DeviceDataLoader(dataloader, device)

    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()

    #####################################################
    # DISCRIMINATOR MODEL 
    #####################################################

    # leakyrelu since letting some neg grad info flow to generator helps
    D = nn.Sequential(
        # in: 3 (color channels) x 128 x 128 (pixel)
        # this size is for a single image, but models processes batches so that dimension 
        # is multiplied by batch_size

        nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
        # stride 2 makes half the size of the pixel every iter
        nn.BatchNorm2d(64), #  batch norm helps to keep gradient uniform
        nn.LeakyReLU(0.2, inplace=True),
        # out: 64 (channels) x 64 x 64 

        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 128 (channels) x32 x 32

        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 256 x 16 x 16

        nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 512 x 8 x 8

        
        nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
        # strid 1, no pading so convolution covers precisely the 4x4 image and output 1
        # out: 1 x 1 x 1

        nn.Flatten(), # reduces to 1d from 3d tensor
        nn.Sigmoid())

    D = to_device(D, device)
    '''nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(1024),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 1024 x 4 x 4'''
    ###################################
    # GENERATOR MODEL
    ################################### 

    # relu since negative info does not help further the discriminator
    # operation is opposite to convolution, we reduce channels while incresing dimensions
    G = nn.Sequential(
        # in: latent_size x 1 x 1
        # it starts from a random vector

        nn.ConvTranspose2d(nz, 512, kernel_size=4, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
        # out: 512 x 4 x 4

        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
        # in inverse conv stride=2 multiplies by 2 instead of reducing
        nn.BatchNorm2d(256),
        nn.ReLU(True),
        # out: 256 x 8 x 8

        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(True),
        # out: 128 x 16 x 16

        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        # out: 64 x 32 x 32
        
        nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
        nn.Tanh() # bring balues to -1,1 normalization as input images
        # out: 3 x 128 x 128
    )

    '''nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(32),
    nn.ReLU(True),
    # out: 32 x 64 x 64'''

    G = to_device(G, device)

 # optim params
    lr = 0.0002
    criterion = nn.BCELoss()
    d_opt = torch.optim.Adam(D.parameters(), lr, betas=(0.5, 0.999)) # betas are empirical params found to work well
    g_opt = torch.optim.Adam(G.parameters(), lr, betas=(0.5, 0.999))

    # just resetting the grad, otherwise will keep adding every iter
    def reset_grad():
        d_opt.zero_grad()
        g_opt.zero_grad()

    # discriminator training fcn
    def train_discriminator(images):
        # Loss for real images
        real_preds  = D(images)
        real_targets = torch.ones(images.size(0), 1).to(device)
        d_loss_real = criterion(real_preds, real_targets)
        real_score = torch.mean(real_preds).item()

        # Loss for fake images
        z = torch.randn(batch_size, nz, 1, 1).to(device)
        fake_images = G(z)
        fake_preds  = D(fake_images)
        fake_labels = torch.zeros(fake_images.size(0), 1).to(device)
        d_loss_fake = criterion(fake_preds, fake_labels)
        fake_score = torch.mean(fake_preds).item()

        # total loss
        d_loss = d_loss_real + d_loss_fake
        # reset grad
        reset_grad()
        # compue gradient
        d_loss.backward()
        # update params
        d_opt.step()

        return d_loss.item(), real_score, fake_score 


    # generator training fcn
    def train_generator():
        # generate fake images and loss computation
        z = torch.randn(batch_size, nz, 1, 1).to(device)
        fake_images = G(z)
        labels = torch.ones(batch_size, 1).to(device) # labeled as true even if fake to train to produce similar to real ones
        g_loss = criterion(D(fake_images), labels)    # loss will be low of discriminator will recognize as true 
        
        reset_grad()
        g_loss.backward()
        g_opt.step()
        
        return g_loss.item()

        
    # dir to output samples
    sample_dir = 'samples/art'
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # define a constant set of images to track evolution of the model
    sample_vectors = torch.randn(batch_size, nz, 1, 1).to(device)

    def save_fake_images(idx, sample_vectors):
        fake_images = G(sample_vectors)
        fake_fname = 'generated_images-{0:0=4d}.png'.format(idx)
        save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=4)

    # save initial images before training
    save_fake_images(0, sample_vectors)

    # training function
    num_epochs = 50
    d_losses, g_losses, real_scores, fake_scores = [], [], [], []

    def fit(num_epochs):
        torch.cuda.empty_cache()

        for epoch in range(num_epochs):
            for real_images, _ in tqdm(dataloader):

                # train discriminator and generator
                d_loss, real_score, fake_score = train_discriminator(real_images)
                g_loss = train_generator()

                # save losses
                d_losses.append(d_loss)
                g_losses.append(g_loss)
                real_scores.append(real_score)
                fake_scores.append(fake_score)
                
            print('Epoch [{}/{}], D_loss: {:.4f}, G_loss: {:.4f}, D_score: {:.2f}, G_score: {:.2f}'.format(
                epoch, num_epochs, d_loss, g_loss, real_score, fake_score))

            # sample and save images
            save_fake_images(epoch+1, sample_vectors) 
        
        return d_losses, g_losses, real_scores, fake_scores

    # running training
    history = fit(num_epochs)


if __name__ == "__main__":
    main()