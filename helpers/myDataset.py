from torch.utils.data import Dataset
import numpy as np
import torch
from os.path import join
from struct import unpack

class TrainData(Dataset):
	def __init__(self,data_name,file_path):

	    if data_name == "sonar":
	        dataset = np.loadtxt(file_path, dtype=str, skiprows=0, delimiter=',')
	        feature = [dataset[i][0:-1] for i in range(len(dataset))]
	        label = [0 if dataset[i][-1]=='R' else 1 for i in range(len(dataset))]
	        
	        train_feature = np.array([feature[i]  for i in range(len(feature)) if i % 3 != 0],dtype=np.float32)
	        train_label = np.array([label[i]  for i in range(len(label)) if i % 3 != 0],dtype=np.int64)

	    elif data_name == "wdbc":
	        dataset = np.loadtxt(file_path, dtype=str, skiprows=0, delimiter=',')
	        feature = [dataset[i][2:] for i in range(len(dataset))]
	        label = [0 if dataset[i][1]=='M' else 1 for i in range(len(dataset))]
	        
	        train_feature = np.array([feature[i]  for i in range(len(feature)) if i % 3 != 0],dtype=np.float32)
	        train_label = np.array([label[i]  for i in range(len(label)) if i % 3 != 0],dtype=np.int64)

	    elif data_name == "soybean":
	        dataset = np.loadtxt(file_path, dtype=str, skiprows=0, delimiter=',')
	        for i in range(len(dataset)):
	            for j in range(len(dataset[i])):
	                if dataset[i][j] == '?':
	                    dataset[i][j] = '-1'
	        feature = [dataset[i][1:] for i in range(len(dataset))]
	        label = []
	        num = 0
	        label_dict = {}
	        for i in range(len(dataset)):
	            if dataset[i][0] in label_dict:
	                label.append(label_dict[dataset[i][0]])
	            else:
	                label_dict[dataset[i][0]] = num
	                label.append(num)
	                num = num + 1

	        train_feature = np.array(feature[0:307],dtype=np.float32)
	        train_label = np.array(label[0:307],dtype=np.int64)

	    elif data_name == "robot" or data_name == "iris":
	        dataset = np.loadtxt(file_path, dtype=str, skiprows=0, delimiter=',')
	        feature = [dataset[i][0:-1] for i in range(len(dataset))]
	        label = []
	        num = 0
	        label_dict = {}
	        for i in range(len(dataset)):
	            if dataset[i][-1] in label_dict:
	                label.append(label_dict[dataset[i][-1]])
	            else:
	                label_dict[dataset[i][-1]] = num
	                label.append(num)
	                num = num + 1

	        train_feature = np.array([feature[i]  for i in range(len(feature)) if i % 3 != 0],dtype=np.float32)
	        train_label = np.array([label[i]  for i in range(len(label)) if i % 3 != 0],dtype=np.int64)

	    self.x=torch.from_numpy(train_feature)
	    self.y=torch.from_numpy(train_label)

	def __getitem__(self,index):
		return self.x[index],self.y[index]

	def __len__(self):
		return self.y.size(0)

class TestData(Dataset):
	def __init__(self,data_name,file_path):

	    if data_name == "sonar":
	        dataset = np.loadtxt(file_path, dtype=str, skiprows=0, delimiter=',')
	        feature = [dataset[i][0:-1] for i in range(len(dataset))]
	        label = [0 if dataset[i][-1]=='R' else 1 for i in range(len(dataset))]
	        
	        val_feature = np.array([feature[i]  for i in range(len(feature)) if i % 3 == 0],dtype=np.float32)
	        val_label = np.array([label[i]  for i in range(len(label)) if i % 3 == 0],dtype=np.int64)

	    elif data_name == "wdbc":
	        dataset = np.loadtxt(file_path, dtype=str, skiprows=0, delimiter=',')
	        feature = [dataset[i][2:] for i in range(len(dataset))]
	        label = [0 if dataset[i][1]=='M' else 1 for i in range(len(dataset))]
	        
	        val_feature = np.array([feature[i]  for i in range(len(feature)) if i % 3 == 0],dtype=np.float32)
	        val_label = np.array([label[i]  for i in range(len(label)) if i % 3 == 0],dtype=np.int64)

	    elif data_name == "soybean":
	        dataset = np.loadtxt(file_path, dtype=str, skiprows=0, delimiter=',')
	        for i in range(len(dataset)):
	            for j in range(len(dataset[i])):
	                if dataset[i][j] == '?':
	                    dataset[i][j] = '-1'
	        feature = [dataset[i][1:] for i in range(len(dataset))]
	        label = []
	        num = 0
	        label_dict = {}
	        for i in range(len(dataset)):
	            if dataset[i][0] in label_dict:
	                label.append(label_dict[dataset[i][0]])
	            else:
	                label_dict[dataset[i][0]] = num
	                label.append(num)
	                num = num + 1
	        
	        val_feature = np.array(feature[307:],dtype=np.float32)
	        val_label = np.array(label[307:],dtype=np.int64)

	    elif data_name == "robot" or data_name == "iris":
	        dataset = np.loadtxt(file_path, dtype=str, skiprows=0, delimiter=',')
	        feature = [dataset[i][0:-1] for i in range(len(dataset))]
	        label = []
	        num = 0
	        label_dict = {}
	        for i in range(len(dataset)):
	            if dataset[i][-1] in label_dict:
	                label.append(label_dict[dataset[i][-1]])
	            else:
	                label_dict[dataset[i][-1]] = num
	                label.append(num)
	                num = num + 1
	        
	        val_feature = np.array([feature[i]  for i in range(len(feature)) if i % 3 == 0],dtype=np.float32)
	        val_label = np.array([label[i]  for i in range(len(label)) if i % 3 == 0],dtype=np.int64)

	    self.x=torch.from_numpy(val_feature)
	    self.y=torch.from_numpy(val_label)

	def __getitem__(self,index):
		return self.x[index],self.y[index]

	def __len__(self):
		return self.y.size(0)

class MnistDataset(Dataset):
	def __init__(self,path,kind):
		assert kind=='train' or kind=='t10k','Unsupported kind '+kind

		#generate full path
		labels_path = join(path,'%s-labels.idx1-ubyte' % kind)
		images_path = join(path,'%s-images.idx3-ubyte' % kind)

		#open files and read
		with open(labels_path,'rb') as lbpath:
			magic,n = unpack('>II',lbpath.read(8))
			y = np.fromfile(lbpath,dtype=np.uint8)
		with open(images_path,'rb') as imgpath:
			magic,num,rows,cols = unpack('>IIII',imgpath.read(16))
			X = np.fromfile(imgpath,dtype=np.uint8).reshape(len(y),1,28,28)

		if kind=='train':
			assert X.shape==(60000,1,28,28),'Data missed partially, expect (60000,1,28,28), but got '+str(X.shape)+' instead'
			assert y.shape==(60000,),'Data missed partially, expect (60000,), but got '+str(y.shape)+' instead'
		else:
			assert X.shape==(10000,1,28,28),'Data missed partially, expect (10000,1,28,28), but got '+str(X.shape)+' instead'
			assert y.shape==(10000,),'Data missed partially, expect (10000,), but got '+str(y.shape)+' instead'

		#convert all to tensors
		self.X = torch.from_numpy(X.astype(np.float32))
		self.y = torch.from_numpy(y.astype(np.int64))

	def __getitem__(self,index):
		return self.X[index],self.y[index]

	def __len__(self):
		return self.y.size(0)
