from torch import nn
from models.BaseModels import *

#small convolutional network for MNIST dataset
class ConvNetMNIST(nn.Module):
	def __init__(self,input_shape,num_feature,num_class,transfered=None):
		super(ConvNetMNIST,self).__init__()

		self.conv = nn.Sequential( #input_shape[0]*input_shape[1]*input_shape[2]
			Conv2dSame(in_channels=input_shape[0],out_channels=32,kernel_size=3),
			nn.ReLU(),
			Conv2dSame(in_channels=32,out_channels=64,kernel_size=3),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Dropout(0.25)
		) #64*(input_shape[1]/2)*(input_shape[2]/2)

		self.fc1 = nn.Sequential( #64*(input_shape[1]/2)*(input_shape[2]/2)
			nn.Linear(64*(input_shape[1]//2)*(input_shape[2]//2),num_feature),
			nn.ReLU(),
			nn.Dropout(0.5)
		) #num_feature

		self.fc2 = nn.Linear(num_feature,num_class)

	def forward(self,x,mode='all'):
		x = self.conv(x)
		x = x.view(x.size(0),-1)
		x = self.fc1(x)
		y = self.fc2(x)
		return y
