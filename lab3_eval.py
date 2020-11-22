import torch
from torch.optim import *
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from helpers.MyDataset import *
from helpers.Metrics import *
from models.ConvNetMNIST import *
from models.ConvNetMNIST_BN import *

#set enumerated class pools
net_classes = [ConvNetMNIST,ConvNetMNIST_BN]
opt_classes = [SGD,RMSprop,Adam]
spec = {
	'ConvNetMNIST'		: '',
	'ConvNetMNIST_BN'	: '_bn'
}
color = {
	'SGD'			: 'r'			,
	'RMSprop'		: 'g'			,
	'Adam'			: 'b'			,
	'ConvNetMNIST'		: {'':'r','_L1':'g'}	,
	'ConvNetMNIST_BN'	: {'':'b','_L1':'black'}
}

if __name__=='__main__':
	index = np.arange(50)

	#accuracy with L1 regularization
	for net_class in net_classes:
		plt.close('all')
		for opt_class in opt_classes:
			accuracies = np.load('results/accuracy_%s%s_L1.npy' %(opt_class.__name__,spec[net_class.__name__]))
			plt.plot(index,accuracies,color[opt_class.__name__])
		plt.savefig('results/accuracy%s_L1.jpg' % spec[net_class.__name__])

	#accuracy without L1 regularization
	for net_class in net_classes:
		plt.close('all')
		for opt_class in opt_classes:
			accuracies = np.load('results/accuracy_%s%s.npy' %(opt_class.__name__,spec[net_class.__name__]))
			plt.plot(index,accuracies,color[opt_class.__name__])
		plt.savefig('results/accuracy%s.jpg' % spec[net_class.__name__])

	#loss with L1 regularization
	for net_class in net_classes:
		plt.close('all')
		for opt_class in opt_classes:
			losses = np.load('results/loss_%s%s_L1.npy' %(opt_class.__name__,spec[net_class.__name__]))
			plt.plot(index,losses,color[opt_class.__name__])
		plt.savefig('results/loss%s_L1.jpg' % spec[net_class.__name__])

	#loss without L1 regularization
	for net_class in net_classes:
		plt.close('all')
		for opt_class in opt_classes:
			losses = np.load('results/loss_%s%s.npy' %(opt_class.__name__,spec[net_class.__name__]))
			plt.plot(index,losses,color[opt_class.__name__])
		plt.savefig('results/loss%s.jpg' % spec[net_class.__name__])

	#accuracy for each optimizer
	for opt_class in opt_classes:
		plt.close('all')
		for net_class in net_classes:
			for regularization in ['','_L1']:
				accuracies = np.load('results/accuracy_%s%s%s.npy' %(opt_class.__name__,spec[net_class.__name__],regularization))
				plt.plot(index,accuracies,color[net_class.__name__][regularization])
		plt.savefig('results/accuracy_%s.jpg' % opt_class.__name__)

	#loss for each optimizer
	for opt_class in opt_classes:
		plt.close('all')
		for net_class in net_classes:
			for regularization in ['','_L1']:
				losses = np.load('results/loss_%s%s%s.npy' %(opt_class.__name__,spec[net_class.__name__],regularization))
				plt.plot(index,losses,color[net_class.__name__][regularization])
		plt.savefig('results/loss_%s.jpg' % opt_class.__name__)

	print('done')
