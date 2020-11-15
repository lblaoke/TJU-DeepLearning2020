import torch
from torch import nn
from sys import argv
from helpers.myDataset import *
from helpers.myfunc import *
from torch.utils.data import DataLoader
from models.ConvNetMNIST import *
import matplotlib.pyplot as plt

#test device status
device_status=torch.cuda.is_available()
if device_status:
	try:
		device_id=free_device_id(argv[1])
	except:
		device_id=0

#initialize model
net=ConvNetMNIST((1,28,28),128,10)
if device_status:
	net=net.to(device_id)

#set hyperparameters
loss_func=nn.CrossEntropyLoss()
opt=torch.optim.Adam(net.parameters(),lr=0.001)
mini_batch=512

#load dataset
trainData=MnistDataset('../shared/datasets/mnist/','train')
testData=MnistDataset('../shared/datasets/mnist/','t10k')

trainLoader=DataLoader(
	dataset=trainData,
	batch_size=mini_batch,
	shuffle=True,
	num_workers=0
)
testLoader=DataLoader(
	dataset=testData,
	batch_size=4096,
	shuffle=False,
	num_workers=0
)

if __name__=='__main__':
	epochs,accuracys,losses=[],[],[]

	#train & test
	for epoch in range(40):
		loss_sum=0.
		net=net.train()
		for _,(x,y) in enumerate(trainLoader):
			if device_status:
				x,y=x.to(device_id),y.to(device_id)

			#calcualte estimated results
			opt.zero_grad()
			y_hat=net(x)

			#calculate loss and propagate back
			loss=loss_func(y_hat,y)
			loss.backward()
			opt.step()
			loss_sum+=mini_batch*loss.item()
		
		net=net.eval()
		positive_n=0
		for _,(x,y) in enumerate(testLoader):
			if device_status:
				x=x.to(device_id)

			#predict
			with torch.no_grad():
				y_hat=net(x)	

			#compare and count
			for i in range(len(y)):
				if torch.argmax(y_hat[i]).item()==y[i].item():
					positive_n+=1

		print('epoch = %d	accuracy = %f' %(epoch,positive_n/testData.__len__()))
		epochs.append(epoch)
		accuracys.append(positive_n/testData.__len__())
		losses.append(loss_sum/60000)

	plt.plot(epochs,accuracys)
	plt.savefig('./results/accuracy.jpg')
	plt.close('all')
	plt.plot(epochs,losses)
	plt.savefig('./results/loss.jpg')

	#save parameters
	torch.save(net.state_dict(),'./results/ConvNetMNIST.pkl')
