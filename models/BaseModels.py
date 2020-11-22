from torch import nn
import torch.nn.functional as F

#convolutional module with identical input and output size
class Conv2dSame(nn.Module):
	def __init__(self,in_channels,out_channels,kernel_size):
		assert type(kernel_size)==int,'Unsupported type '+str(type(kernel_size))+' for kernel_size'

		super(Conv2dSame,self).__init__()

		bound = kernel_size-1
		bound_l = bound//2
		bound_r = bound-bound_l

		self.padding = nn.ReplicationPad2d((bound_l,bound_r,bound_l,bound_r))
		self.conv = nn.Conv2d(
			in_channels	= in_channels	,
			out_channels	= out_channels	,
			kernel_size	= kernel_size
		)

	def forward(self,x):
		x = self.padding(x)
		y = self.conv(x)
		return y

#Conv2dSame+ReLU+BatchNorm2d
class Conv2dSame_ReLU_BN(nn.Module):
	def __init__(self,in_channels,out_channels,kernel_size):
		assert type(kernel_size)==int,'Unsupported type '+str(type(kernel_size))+' for kernel_size'

		super(Conv2dSame_ReLU_BN,self).__init__()

		bound = kernel_size-1
		bound_l = bound//2
		bound_r = bound-bound_l

		self.padding = nn.ReplicationPad2d((bound_l,bound_r,bound_l,bound_r))
		self.conv = nn.Conv2d(
			in_channels	= in_channels	,
			out_channels	= out_channels	,
			kernel_size	= kernel_size
		)
		self.bn = nn.BatchNorm2d(out_channels)

	def forward(self,x):
		x = self.padding(x)
		x = self.conv(x)
		x = F.relu(x)
		y = self.bn(x)
		return y
