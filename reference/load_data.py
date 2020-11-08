import torch
import numpy as np

def load_data(data_name, file_path):
    if data_name == "sonar":
        #dataset = np.loadtxt("/home/ubuntu/zhangli/datasets/datasets/sonar_binary/sonar.all-data", dtype=str, skiprows=0, delimiter=',')
        dataset = np.loadtxt(file_path, dtype=str, skiprows=0, delimiter=',')
        feature = [dataset[i][0:-1] for i in range(len(dataset))]
        label = [0 if dataset[i][-1]=='R' else 1 for i in range(len(dataset))]
        train_feature = torch.tensor(np.matrix([feature[i]  for i in range(len(feature)) if i % 3 != 0]).astype(float),dtype=torch.float32)
        val_feature = torch.tensor(np.matrix([feature[i]  for i in range(len(feature)) if i % 3 == 0]).astype(float),dtype=torch.float32)
        train_label = torch.tensor([label[i]  for i in range(len(label)) if i % 3 != 0])
        val_label = torch.tensor([label[i]  for i in range(len(label)) if i % 3 == 0])
    elif data_name == "wdbc":
        #dataset = np.loadtxt("/home/ubuntu/zhangli/datasets/datasets/wdbc_binary/wdbc.data", dtype=str, skiprows=0, delimiter=',')
        dataset = np.loadtxt(file_path, dtype=str, skiprows=0, delimiter=',')
        feature = [dataset[i][2:] for i in range(len(dataset))]
        label = [0 if dataset[i][1]=='M' else 1 for i in range(len(dataset))]
        train_feature = torch.tensor(np.matrix([feature[i]  for i in range(len(feature)) if i % 3 != 0]).astype(float),dtype=torch.float32)
        val_feature = torch.tensor(np.matrix([feature[i]  for i in range(len(feature)) if i % 3 == 0]).astype(float),dtype=torch.float32)
        train_label = torch.tensor([label[i]  for i in range(len(label)) if i % 3 != 0])
        val_label = torch.tensor([label[i]  for i in range(len(label)) if i % 3 == 0])
    elif data_name == "soybean":
        #dataset = np.loadtxt("/home/ubuntu/zhangli/datasets/datasets/soybean_multi/soybean-large.data", dtype=str, skiprows=0, delimiter=',')
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
        train_feature = torch.tensor(np.matrix(feature[0:307]).astype(float),dtype=torch.float32)
        val_feature = torch.tensor(np.matrix(feature[307:]).astype(float),dtype=torch.float32)
        train_label = torch.tensor(label[0:307])
        val_label = torch.tensor(label[307:])
    elif data_name == "robot" or data_name == "iris":
        #dataset = np.loadtxt("/home/ubuntu/zhangli/datasets/datasets/robot_multi/sensor_readings_24.data", dtype=str, skiprows=0, delimiter=',')
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
        train_feature = torch.tensor(np.matrix([feature[i]  for i in range(len(feature)) if i % 3 != 0]).astype(float),dtype=torch.float32)
        val_feature = torch.tensor(np.matrix([feature[i]  for i in range(len(feature)) if i % 3 == 0]).astype(float),dtype=torch.float32)
        train_label = torch.tensor([label[i]  for i in range(len(label)) if i % 3 != 0])
        val_label = torch.tensor([label[i]  for i in range(len(label)) if i % 3 == 0])
    train_data = torch.utils.data.TensorDataset(train_feature,train_label)
    val_data = torch.utils.data.TensorDataset(val_feature,val_label)
    return train_data, val_data

