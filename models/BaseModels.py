from torch import nn

#convolutional module with identical input and output size
class Conv2dSame(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):
        super(Conv2dSame,self).__init__()

        ka = kernel_size//2
        kb = ka-1 if kernel_size%2==0 else ka

        self.net = nn.Sequential(
            nn.ReflectionPad2d((ka,kb,ka,kb)),
            nn.Conv2d(in_channels,out_channels,kernel_size)
        )

    def forward(self,x):
        return self.net(x)
