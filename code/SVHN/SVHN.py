import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

batch_size = 16
epochs = 5
transforms = transforms.Compose([
    transforms.Resize((128,128)), #(64,128)(128,128)
    # transforms.CenterCrop((60,120)),#(60,120)
    transforms.ColorJitter(0.3,0.3,0.2),#(0.3,0.3,0.2)(0.5,0.5,0.5)
    transforms.RandomRotation(5), #(5)(20)
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.465,0.406],[0.229,0.224,0.225])
])

train_set = datasets.SVHN("data_svhn","train",download=True,transform=transforms)
test_set = datasets.SVHN("data_svhn","test",download=True,transform=transforms)


train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=True)

print("下载完成")

class SVHN_Net(nn.Module):
    def __init__(self):
        super(SVHN_Net,self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3,16,3,2), #64*63*63
            nn.ReLU(),
            nn.MaxPool2d(2), #64*31*31
            nn.Conv2d(16,32,3), #64 * 29 * 29
            nn.ReLU(),
            nn.MaxPool2d(2), #64*14*14
            nn.Conv2d(32,64,2), #64 * 13 * 13
            nn.ReLU(),
            nn.Conv2d(64,128,2), #128 * 12 * 12
            nn.ReLU(),
            nn.MaxPool2d(2), #128 * 6 * 6
            nn.Conv2d(128,128,3), #128 *4 *4
            nn.ReLU(),
            nn.MaxPool2d(2), #128 *2 *2


            # nn.Conv2d(3,16,3,2), #16*29*59
            # nn.ReLU(),
            # nn.MaxPool2d(2), #16*14*29
            # nn.Conv2d(16,32,3,2), #32*6*14
            # nn.ReLU(),
            # nn.MaxPool2d(2), #32*3*7
            # nn.Conv2d(32,32,2), #32*2*6
            # nn.ReLU(),
            # nn.MaxPool2d(2) # 32*1*3
        )


        self.fc1 = nn.Linear(128 *2 *2,128)
        self.fc2 = nn.Linear(128,11)

        # self.fc1 = nn.Linear(32 * 1 * 3, 11)
        # self.fc2 = nn.Linear(32 * 3 * 7, 11)
        # self.fc3 = nn.Linear(32 * 3 * 7, 11)
        # self.fc4 = nn.Linear(32 * 3 * 7, 11)
        # self.fc5 = nn.Linear(32 * 3 * 7, 11)

    def forward(self,x):
        # print(x.shape)
        cnn_res = self.cnn(x)
        # print(cnn_res.shape) #16*32*1*3
        cnn_res = cnn_res.view(cnn_res.size(0),-1)
        # print(cnn_res.shape) #16*96
        f1 = self.fc1(cnn_res)
        f1 = self.fc2(f1)
        # f2 = self.fc2(cnn_res)
        # f3 = self.fc3(cnn_res)
        # f4 = self.fc4(cnn_res)
        # f5 = self.fc5(cnn_res)

        return f1

model = SVHN_Net()
optimizer = optim.Adam(model.parameters(),0.001)
criterion = nn.CrossEntropyLoss()
def train_model(model, train_loader,optimizer ,epoch):
    #模型训练
    model.train()
    for batch_index,(data,target) in enumerate(train_loader):
        #部署到device上
        # data,target = data.to(device),target.to(device)
        # print(data)
        # print(target)
        # print(target)
        #梯度初始化为0
        optimizer.zero_grad()
        #预测
        output = model(data)
        #计算损失
        loss = criterion(output,target)
        #找到概率值最大的下标
        # pred = output.max(1,keepdim=True)
        #反向传播
        loss.backward()
        optimizer.step()
        if batch_index % 200 == 0:
            print("Train Epoch : {} \t Loss : {:.6f}".format(epoch,loss.item()))


def test_model(model,test_loader):
    #模型验证
    model.eval()
    #正确率
    correct = 0.0
    #测试损失
    test_loss = 0.0
    with torch.no_grad():#不会计算梯度也不会进行反向传播
        for data,target in test_loader:
            # data,target = data.to(device),target.to(device)
            #测试数据
            output = model(data)
            #计算测试损失
            test_loss+=F.cross_entropy(output,target).item()
            #找到概率值最大的下标
            pred = output.max(1,keepdim=True)[1] #值 索引
            #pred = torch.max(output,dim=1)
            #pred = output.argmax(dim=1)
            #累计正确率
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print("Test ---- Average loss : {:.4f},Accuracy : {:.3f}\n".format(test_loss,100.0*correct/len(test_loader.dataset)))

for epoch in range(epochs+1):
    train_model(model,train_loader,optimizer,epoch)
    test_model(model,test_loader)