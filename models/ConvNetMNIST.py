from torch import nn

class ConvNetMNIST(nn.Module):
    def __init__(self):
        super(ConvNetMNIST,self).__init__()

        self.conv=nn.Sequential( #1*28*28
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        ) #64*12*12

        self.fc=nn.Sequential( #9216
            nn.Linear(9216,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,10)
        ) #10
        
    def forward(self,x):
        x=self.conv(x)
        x=x.view(x.size(0),-1)
        y=self.fc(x)

        return y
