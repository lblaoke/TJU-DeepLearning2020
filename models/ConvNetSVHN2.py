from torch import nn
from models.BaseModels import *

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

#small convolutional network for SVHN dataset (format 2)
class ConvNetSVHN2(nn.Module):
	def __init__(self,num_feature):
		super(ConvNetSVHN2,self).__init__()

		self.conv = nn.Sequential( #3*32*32
			Conv2dSame_BN_ReLU(3,32,3)	,
			Conv2dSame_BN_ReLU(32,32,3)		,
			nn.MaxPool2d(2)				,
			nn.Dropout(0.3)				,
			Conv2dSame_BN_ReLU(32,64,3)		,
			Conv2dSame_BN_ReLU(64,64,3)		,
			nn.MaxPool2d(2)				,
			nn.Dropout(0.3)				,
			Conv2dSame_BN_ReLU(64,128,3)		,
			Conv2dSame_BN_ReLU(128,128,3)		,
			nn.MaxPool2d(2)				,
			nn.Dropout(0.3)
		) #128*(32/8)*(32/8)

		self.eca = eca_layer(128)

		self.fc = nn.Sequential( #128*(32/8)*(32/8)
			nn.Linear(128,num_feature)	,
			nn.ReLU()					,
			nn.Dropout(0.3)					,
			nn.Linear(num_feature,10)
		) #10

	def forward(self,x):
		x = self.conv(x)
		#x = self.eca(x)
		#x = x.flatten(1,-1)
		x = x.mean(-1).mean(-1)
		x = self.fc(x)

		return x
