import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from helpers.MyDataset import *

trainData = MnistDataset('../datasets/mnist/','train')
trainLoader=DataLoader(
	dataset		= trainData	,
	batch_size	= 16		,
	shuffle		= True		,
	num_workers	= 0		,
	drop_last	= False
)

img_shape = (1, 28, 28)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(10, 10)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(100 + 10, 128, normalize=False),
            *block(128, 512),
            *block(512, 512),
            *block(512, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(10, 10)

        self.model = nn.Sequential(
            nn.Linear(10 + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity

ad_loss = torch.nn.MSELoss().cuda()
generator = Generator().cuda()
discriminator = Discriminator().cuda()

opt_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5,0.999))
opt_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5,0.999))


FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

n_epochs = 500
def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, 100))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "results/cgan%d.jpg" % batches_done, nrow=n_row, normalize=True)


for epoch in range(n_epochs):
    for i,(x,y) in enumerate(trainLoader):

        valid = Variable(FloatTensor(x.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(x.shape[0], 1).fill_(0.0), requires_grad=False)

        real_img=Variable(x.type(FloatTensor))
        y = Variable(y.type(LongTensor))

        opt_G.zero_grad()

        z = Variable(FloatTensor(np.random.normal(0, 1, (x.shape[0], 100))))

        fake_img = Variable(LongTensor(np.random.randint(0, 10, x.shape[0])))

        gen = generator(z,fake_img)
        validity = discriminator(gen,fake_img)
        g_loss = ad_loss(validity,valid)

        g_loss.backward()
        opt_G.step()

        opt_D.zero_grad()
        valid_real = discriminator(real_img,y)
        valid_loss = ad_loss(valid_real,valid)

        valid_fake = discriminator(gen.detach(),fake_img)
        fake_loss = ad_loss(valid_fake,fake)

        d_loss = (valid_loss+fake_loss)/2

        d_loss.backward()
        opt_D.step()

        if i==0:
            print('epoch = %d, loss_G = %f, loss_D = %f' %(epoch,d_loss.item(),g_loss.item()))

        batches_done = epoch * len(trainLoader) + i
        if batches_done % 1000 == 0:
            sample_image(n_row=10, batches_done=batches_done)
