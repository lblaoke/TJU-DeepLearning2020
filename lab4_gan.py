import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from helpers.MyDataset import *

#test device status
device_status = torch.cuda.is_available()
if device_status:
	device_id = 0
	torch.backends.cudnn.enabled = True
	torch.backends.cudnn.benchmark = True

#initialize model
G = nn.Sequential(
	nn.Linear(100,128)		,
	nn.LeakyReLU(0.2,inplace=True)	,

	nn.Linear(128,256)		,
	nn.BatchNorm1d(256,0.8)		,
	nn.LeakyReLU(0.2,inplace=True)	,

	nn.Linear(256,512)		,
	nn.BatchNorm1d(512,0.8)		,
	nn.LeakyReLU(0.2,inplace=True)	,

	nn.Linear(512,1024)		,
	nn.BatchNorm1d(1024,0.8)	,
	nn.LeakyReLU(0.2,inplace=True)	,

	nn.Linear(1024,1*28*28)		,
	nn.Hardsigmoid()		,
	nn.Unflatten(1,(1,28,28))
)
D = nn.Sequential(
	nn.Flatten()			,
	nn.Linear(1*28*28,512)		,
	nn.LeakyReLU(0.2,inplace=True)	,
	nn.Linear(512,256)		,
	nn.LeakyReLU(0.2,inplace=True)	,
	nn.Linear(256,1)		,
	nn.Hardsigmoid()
)
if device_status:
	G = G.to(device_id)
	D = D.to(device_id)

#set hyperparameters
mini_batch = 128
opt_G = torch.optim.Adam(G.parameters(),lr=0.0001, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(),lr=0.0001, betas=(0.5, 0.999))
loss_func = nn.BCELoss()
if device_status:
	loss_func = loss_func.to(device_id)

#load dataset
trainData = MnistDataset('D://datasets/mnist/','t10k')
trainLoader=DataLoader(
	dataset		= trainData	,
	batch_size	= mini_batch	,
	shuffle		= True		,
	num_workers	= 0		,
	drop_last	= True
)

if __name__=='__main__':

	#ground truth
	true = torch.ones(mini_batch,1,dtype=torch.float32,requires_grad=False)
	false = torch.zeros(mini_batch,1,dtype=torch.float32,requires_grad=False)
	test_z = torch.randn(100,100)
	if device_status:
		true,false = true.to(device_id),false.to(device_id)
		test_z = test_z.to(device_id)

	losses = []
	for epoch in range(101):
		loss_G,loss_D = 0.,0.
		for _,(real,_) in enumerate(trainLoader):
			z = torch.randn(mini_batch,100)
			if device_status:
				real = real.to(device_id)
				z = z.to(device_id)

			#train discriminator
			if loss_G<100:
				opt_D.zero_grad()
				loss_real = loss_func(D(real),true)
				loss_fake = loss_func(D(G(z)),false)
				loss = (loss_real+loss_fake)/2.
				loss.backward(retain_graph=True)
				opt_D.step()
				loss_D += loss.item()

			#train generator
			opt_G.zero_grad()
			loss = loss_func(D(G(z)),true)
			loss.backward(retain_graph=False)
			opt_G.step()
			loss_G += loss.item()

		print('epoch = %d, loss_G = %f, loss_D = %f' %(epoch,loss_G,loss_D))
		losses.append(loss_D)

		#test
		'''
		if epoch%5==0:
			fake = G(test_z)

			if device_status:
				fake = fake.cpu()
			fake = fake.detach().numpy()

			plot = []
			for i in range(10):
				row = []
				for j in range(10):
					row.append(fake[i+10*j])
				row = np.concatenate(row,axis=2)
				plot.append(row)
			plot = np.concatenate(plot,axis=1)
			plot = (plot*255.).astype(np.uint8)
			plot = plot.reshape(280,280)

			cv2.imwrite('results/gan%d.jpg' % epoch,plot)
		'''
	index = np.arange(101)
	plt.plot(index,losses)
	plt.show()
