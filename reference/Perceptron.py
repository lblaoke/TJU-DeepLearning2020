import torch
import torch.nn as nn
import numpy as np
from load_data import load_data

#sonar  Attributes : 60  Classes : 2
#wdbc   Attributes : 30	 Classes : 2

dataset_path ={"sonar":"/home/ubuntu/zhangli/datasets/datasets/sonar_binary/sonar.all-data",
               "wdbc":"/home/ubuntu/zhangli/datasets/datasets/wdbc_binary/wdbc.data",
              }
class Perceptron(nn.Module):
    def __init__(self, in_dim=2, out_dim=2):
        super(Perceptron, self).__init__()
        self.net = nn.Linear(in_dim, out_dim)
        #参数初始化
        for params in self.net.parameters():
            nn.init.normal_(params, mean=0, std=0.01)
    #输入数据在模型中前向传播的计算过程
    def forward(self, x):
        x = self.net(x)
        return x
#训练权重
def train(train_loader, model, criterion, optimizer):
    model.train()
    for i, (feature, label) in enumerate(train_loader):
        output = model(feature)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#验证结果       
def validate(val_loader, model, criterion):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i, (feature, label) in enumerate(val_loader):
            output = model(feature)
            #计算预测准确率
            _, pred = output.topk(1, 1, True, True)
            correct += pred.eq(label.view(-1, 1)).sum(0, keepdim=True)
    return correct[0] 
        

if __name__ == '__main__':
    model = Perceptron(in_dim=30,out_dim=2)
    epoch = 100
    batch_size = 20
    learning_rate = 0.05
    #损失函数
    criterion = nn.CrossEntropyLoss()
    #优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    #读入训练数据
    train_data, val_data = load_data("wdbc",dataset_path["wdbc"])
    #将训练数据分成一个个batch
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    for i in range(epoch):
        train(train_loader, model, criterion, optimizer)
        correct = validate(val_loader, model, criterion)
        print("epoch:\t",i,"acc:\t",correct*100.0/len(val_data))
        
    

    
    
