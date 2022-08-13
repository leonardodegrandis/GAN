import os
from IPython.display import Image
from charset_normalizer import from_fp
import torch
from torchvision.utils import save_image
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

# images are normalized around -1,1 to work better in GANs
mnist = MNIST(root= 'dataset', 
              train=True, 
              download=True, 
              transform=Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))]))

img, label = mnist[0]

# define a fcn to denormalize the images before plotting them
def denorm(x):
    out = (x + 1) / 2 # goes back to 0,1 range
    return out.clamp(0, 1) # makes sure that every value is inside the range

'''
img_denorm = denorm(img)
plt.imshow(img_denorm[0], cmap='gray')
plt.show()
'''
# init DataLoader
batch_size = 300
data_loader = DataLoader(mnist, batch_size, shuffle=True)

# init device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# DISCRIMINATOR MODEL 
image_size = 784
hidden_size = 256

# leakyrelu since letting some neg grad info flow to generator helps
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2), 
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid()
    )

D.to(device)

# GENERATOR MODEL 
latent_size = 64

# relu since negative info does not help further the discriminator
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh()
    )

G.to(device)

# optim params
lr = 0.0002
criterion = nn.BCELoss()
d_opt = torch.optim.Adam(D.parameters(), lr)
g_opt = torch.optim.Adam(G.parameters(), lr)

# just resetting the grad, otherwise will keep adding every iter
def reset_grad():
    d_opt.zero_grad()
    g_opt.zero_grad()

# discriminator training fcn
def train_discriminator(images):
    # Loss for real images
    out = D(images)
    real_labels = torch.ones(batch_size, 1).to(device)
    d_loss_real = criterion(out, real_labels)
    real_score = out

    # Loss for fake images
    z = torch.randn(batch_size, latent_size).to(device)
    fake_images = G(z)
    out = D(fake_images)
    fake_labels = torch.zeros(batch_size, 1).to(device)
    d_loss_fake = criterion(out, fake_labels)
    fake_score = out

    # total loss
    d_loss = d_loss_real + d_loss_fake
    # reset grad
    reset_grad()
    # compue gradient
    d_loss.backward()
    # update params
    d_opt.step()

    return d_loss, real_score, fake_score 

# generator training fcn
def train_generator():
    # generate fake images and loss computation
    z = torch.randn(batch_size, latent_size).to(device)
    fake_images = G(z)
    labels = torch.ones(batch_size, 1).to(device) # labeled as true even if fake to train to produce similar to real ones
    g_loss = criterion(D(fake_images), labels)    # loss will be low of discriminator will recognize as true 
    
    reset_grad()
    g_loss.backward()
    g_opt.step()
    
    return g_loss, fake_images

    
# dir to output samples
sample_dir = 'samples/mnist'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# define a constant set of images to track evolution of the model
sample_vectors = torch.randn(batch_size, latent_size).to(device)

def save_fake_images(idx):
    fake_images = G(sample_vectors)
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    fake_fname = 'fake_images-{0:0=4d}.png'.format(idx)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=10)

# save initial images before training
save_fake_images(0)

# training function
num_epochs = 200
total_step = len(data_loader)
d_losses, g_losses, real_scores, fake_scores = [], [], [], []

for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        # load a batch and transform to vectors
        images = images.reshape(batch_size, -1).to(device)

        # train discriminator and generator
        d_loss, real_score, fake_score = train_discriminator(images)
        g_loss, fake_images = train_generator()

        # save losses
        d_losses.append(d_loss)
        g_losses.append(g_loss)
        real_scores.append(real_score.mean().item())
        fake_scores.append(fake_score.mean().item())
        
    print('Epoch [{}/{}], D_loss: {:.4f}, G_loss: {:.4f}, D_score: {:.2f}, G_score: {:.2f}'.format(
        epoch, num_epochs, d_loss.item(), g_loss.item(), real_score.mean().item(), fake_score.mean().item()))

    # sample and save images
    save_fake_images(epoch+1) 