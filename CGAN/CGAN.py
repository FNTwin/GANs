import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
import matplotlib.pyplot as plt 
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

image_dimension=(1,28,28)

class Generator(nn.Module):
    def __init__(self, latent_dimension, labels):
        super(Generator, self).__init__()
        def fc_block(in_f, out_f, normalize=True):
            layers = [nn.Linear(in_f, out_f)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_f, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.embedding=nn.Embedding(labels,labels)
        self.model = nn.Sequential(
            *fc_block(latent_dimension, 128, normalize=False),
            *fc_block(128, 256, normalize=True),
            *fc_block(256, 512, normalize=True),
            *fc_block(512, 1024, normalize=True),
            nn.Linear(1024, int(np.prod(image_dimension))),
            nn.Tanh())
       
    def forward(self, z, labels):
    	label_input=self.embedding(labels)
        x=torch.cat([z,label_input], 1)
        img = self.model(x)
        img = img.view(img.size(0), *image_dimension)
        return img
   
class Discriminator(nn.Module):
    def __init__(self, labels):
        super(Discriminator,self).__init__()
        self.embedding=nn.Embedding(labels,labels)
        self.fc1=nn.Linear(int(np.prod(image_dimension))+labels, 512)
        self.fc2=nn.Linear(512, 256)
        self.fc3= nn.Linear(256, 1)
        self.sigmoid=nn.Sigmoid()
       
       
    def forward(self, img, labels):
        img_flat=img.view(img.size(0), -1)
        label_input=self.embedding(labels)
        x=torch.cat([img_flat,label_input], 1) 
        x=F.leaky_relu( self.fc1(x), 0.2)
        x=self.sigmoid(self.fc3(F.leaky_relu( self.fc2(x), 0.2)))
        return x

class CGAN():
	def __init__(self, epochs, latent_dimension, labels, lr, bs, gpu=True):
		self.g=Generator(latent_dimension, labels)
		self.d=Discriminator(labels)
		self.loss=torch.nn.BCELoss()
		self.optimizer_g=torch.optim.Adam(self.g.parameters(),lr,bs)
		self.optimizer_d=torch.optim.Adam(self.d.parameters(),lr,bs)
		self.epochs=epochs
		self.latent_dim=latent_dimension
		self.labels_number=labels
		self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu") if gpu else torch.device("cpu")

	def train(self,dataloader):
		d_l,g_l=[],[]
		Tensor = torch.cuda.FloatTensor if device is "cuda" else torch.FloatTensor
		for epoch in range(self.epochs):
		    for count, (j,labels) in enumerate(dataloader):
		        #Generate input
		        z= Variable(Tensor(np.random.normal(0, 1, (j.shape[0], self.latent_dim))))
		        fake_label=Variable(torch.LongTensor(np.random.randint(0, 10, j.size(0)))).to(device)
        		valid_label=labels.to(device)
		       
		        #Generator training (zero gradient, predict, backpropagation, adjust weight
		        self.optimizer_g.zero_grad()
		        fake_images=self.g(z.to(self.device), fake_label)
		        valid=self.d(fake_images.to(self.device), fake_label)
		        g_loss=self.loss(valid, Variable(torch.ones(j.size(0))).unsqueeze(1).to(self.device))
		        g_loss.backward()
		        self.optimizer_g.step()
		       
		        #Discriminator training
		        self.optimizer_d.zero_grad()
		        real= self.loss(self.d(j.to(self.device), valid_label), Variable(torch.ones(j.size(0))).unsqueeze(1).to(self.device))
		        fake= self.loss(self.d(fake_images.detach().to(self.device), fake_label),Variable(torch.zeros(j.size(0))).unsqueeze(1).to(self.device))
		        d_loss=(real+fake)/2
		        d_loss.backward()
		        self.optimizer_d.step()
		      
		        d_l.append(d_loss)
		        g_l.append(g_loss)
		       
		    print(
		            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
		            % (epoch, self.epochs, count, len(dataloader), d_loss.item(), g_loss.item())
		        )

def load_mnist(drop_last=True):
	dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=64,
    shuffle=True,
    drop_last=drop_last
    )
    return dataloader

cgan=CGAN(epochs=200,
	      latent_dimension=100,
	      labels=10,
          lr=0.0002,
          bs=(0.5,0.999))

cgan.train(load_mnist())
