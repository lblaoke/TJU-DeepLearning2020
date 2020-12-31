import torch
from torch.utils.data import DataLoader
from helpers.MyDataset import *
from models.ConvNetSVHN2 import *
from helpers.Metrics import *

#test device status
if torch.cuda.is_available():
	device_id = 0
	torch.backends.cudnn.enabled = True
	torch.backends.cudnn.benchmark = True
else:
	device_id = None

#initialize model
net = ConvNetSVHN2(128).to(device_id)

#set hyperparameters
loss_func = nn.CrossEntropyLoss().to(device_id)
opt = torch.optim.SGD(net.parameters(),lr=0.001,momentum=0.9,weight_decay=0.0005)
load_batch,train_batch = 1024,256

#load dataset
trainData = Svhn2Dataset('D://datasets/svhn-format2/','train')
testData = Svhn2Dataset('D://datasets/svhn-format2/','t10k')

trainLoader = DataLoader(
	dataset		= trainData	,
	batch_size	= load_batch	,
	shuffle		= True		,
	num_workers	= 0		,
	drop_last	= True
)
testLoader = DataLoader(
	dataset		= testData	,
	batch_size	= load_batch	,
	shuffle		= False		,
	num_workers	= 0		,
	drop_last	= False
)

if __name__=='__main__':
	for epoch in range(5):
		net = net.train()
		for _,(X,y) in enumerate(trainLoader):
			X,y=X.to(device_id),y.to(device_id)

			batch = 0
			while batch<load_batch:
				X_batch,y_batch = X[batch:batch+train_batch],y[batch:batch+train_batch]

				opt.zero_grad()
				y_batch_hat = net(X_batch)
				loss = loss_func(y_batch_hat,y_batch)
				loss.backward()
				opt.step()

				batch += train_batch

		y_pred = torch.empty(0,10,dtype=torch.float32)
		y_true = testData.y
		net = net.eval()
		for _,(X,y) in enumerate(testLoader):
			X = X.to(device_id)
			with torch.no_grad():
				y_hat = net(X)
			y_hat = y_hat.cpu()
			y_pred = torch.cat([y_pred,y_hat],dim=0)
		y_pred,y_true = y_pred.numpy(),y_true.numpy()
		print(accuracy(y_pred,y_true,task='multi-classification'))
