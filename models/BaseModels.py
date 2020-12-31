from torch import nn
import torch.nn.functional as F

#convolutional module with identical input and output size
class Conv2dSame(nn.Sequential):
	def __init__(self,in_channels,out_channels,kernel_size):
		assert type(kernel_size)==int,'Unsupported type '+str(type(kernel_size))+' for kernel_size'

		bound = kernel_size-1
		bound_l = bound//2
		bound_r = bound-bound_l

		super(Conv2dSame,self).__init__(
			nn.ReplicationPad2d((bound_l,bound_r,bound_l,bound_r)),
			nn.Conv2d(in_channels,out_channels,kernel_size)
		)

#Conv2dSame+ReLU+BatchNorm2d
class Conv2dSame_BN_ReLU(nn.Sequential):
	def __init__(self,in_channels,out_channels,kernel_size):
		assert type(kernel_size)==int,'Unsupported type '+str(type(kernel_size))+' for kernel_size'

		super(Conv2dSame_BN_ReLU,self).__init__(
			Conv2dSame(in_channels,out_channels,kernel_size),
			nn.BatchNorm2d(out_channels)			,
			nn.ReLU()
		)
