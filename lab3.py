import torch
from torch.optim import *
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from helpers.MyDataset import *
from helpers.Metrics import *
from models.ConvNetMNIST import *
from models.ConvNetMNIST_BN import *

#test device status
device_status = torch.cuda.is_available()
device_id = 2

#set enumerated class pools
net_classes = [ConvNetMNIST,ConvNetMNIST_BN]
opt_classes = [SGD,RMSprop,Adam]
spec = {
	'ConvNetMNIST'		: '',
	'ConvNetMNIST_BN'	: '_bn'
}

#set hyperparameters
loss_func = nn.CrossEntropyLoss()
mini_batch = 128
lamda = 0.0001

#load dataset
trainSet = MnistDataset('../datasets/mnist/','train')
testSet = MnistDataset('../datasets/mnist/','t10k')
trainLoader=DataLoader(
	dataset		= trainSet	,
	batch_size	= mini_batch	,
	shuffle		= True		,
	num_workers	= 0
)
testLoader=DataLoader(
	dataset		= testSet	,
	batch_size	= mini_batch	,
	shuffle		= False		,
	num_workers	= 0
)

if __name__=='__main__':

	#with L1 regularization
	for net_class in net_classes:
		for opt_class in opt_classes:

			#instantiate model and optimizer
			net = net_class(
				input_shape	= (1,28,28)	,
				num_feature	= 128		,
				num_class	= 10
			)
			opt = opt_class(net.parameters(),lr=0.001)
			if device_status:
				net = net.to(device_id)

			accuracies,losses = [],[]
			for _ in tqdm(range(50),ncols=70):

				#train
				loss_sum = 0.
				net = net.train()
				for _,(X,y) in enumerate(trainLoader):
					if device_status:
						X,y = X.to(device_id),y.to(device_id)
					opt.zero_grad()
					y_hat=net(X)
					regularization = 0.
					for param in net.parameters():
						regularization += torch.sum(abs(param))
					loss=loss_func(y_hat,y)+lamda*regularization
					loss.backward()
					opt.step()
					loss_sum += loss.item()*len(y)
				losses.append(loss_sum)

				#test
				y_pred = torch.empty(0,10,dtype=torch.float32)
				y_true = testSet.y
				net=net.eval()
				for _,(X,y) in enumerate(testLoader):
					if device_status:
						X = X.to(device_id)
					with torch.no_grad():
						y_hat = net(X)
					if device_status:
						y_hat = y_hat.cpu()
					y_pred = torch.cat([y_pred,y_hat],dim=0)
				y_pred,y_true = y_pred.numpy(),y_true.numpy()
				accuracies.append(accuracy(y_pred,y_true,task='multi-classification'))
			accuracies = np.array(accuracies,dtype=np.float32)
			losses = np.array(losses,dtype=np.float32)
			np.save('results/accuracy_%s%s_L1.npy' %(opt_class.__name__,spec[net_class.__name__]),accuracies)
			np.save('results/loss_%s%s_L1.npy' %(opt_class.__name__,spec[net_class.__name__]),losses)

	#without L1 regularization
	for net_class in net_classes:
		for opt_class in opt_classes:

			#instantiate model and optimizer
			net = net_class(
				input_shape	= (1,28,28)	,
				num_feature	= 128		,
				num_class	= 10
			)
			opt = opt_class(net.parameters(),lr=0.001)
			if device_status:
				net = net.to(device_id)

			accuracies,losses = [],[]
			for _ in tqdm(range(50),ncols=70):

				#train
				loss_sum = 0.
				net = net.train()
				for _,(X,y) in enumerate(trainLoader):
					if device_status:
						X,y = X.to(device_id),y.to(device_id)
					opt.zero_grad()
					y_hat=net(X)
					loss=loss_func(y_hat,y)
					loss.backward()
					opt.step()
					loss_sum += loss.item()*len(y)
				losses.append(loss_sum)

				#test
				y_pred = torch.empty(0,10,dtype=torch.float32)
				y_true = testSet.y
				net=net.eval()
				for _,(X,y) in enumerate(testLoader):
					if device_status:
						X = X.to(device_id)
					with torch.no_grad():
						y_hat = net(X)
					if device_status:
						y_hat = y_hat.cpu()
					y_pred = torch.cat([y_pred,y_hat],dim=0)
				y_pred,y_true = y_pred.numpy(),y_true.numpy()
				accuracies.append(accuracy(y_pred,y_true,task='multi-classification'))
			accuracies = np.array(accuracies,dtype=np.float32)
			losses = np.array(losses,dtype=np.float32)
			np.save('results/accuracy_%s%s.npy' %(opt_class.__name__,spec[net_class.__name__]),accuracies)
			np.save('results/loss_%s%s.npy' %(opt_class.__name__,spec[net_class.__name__]),losses)

	print('done')
