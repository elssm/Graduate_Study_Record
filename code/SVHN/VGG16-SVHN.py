import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from torchvision import models

batch_size = 64
epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


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


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), #64*128*128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), #64*128*128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2), #64*64*64
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), #128*64*64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), #128 * 64 * 64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2) #128 *32 *32
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 32 * 128, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(tensor=m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(tensor=m.bias, val=0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(tensor=m.weight, val=1)
                nn.init.constant_(tensor=m.bias, val=0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(tensor=m.weight, mean=0, std=0.01)
                nn.init.constant_(tensor=m.bias, val=0)


model = VGGNet()
model = model.to(device)

optimizer = optim.Adam(model.parameters(),0.001)
criterion = nn.CrossEntropyLoss()
def train_model(model, train_loader,optimizer ,epoch):
    #模型训练
    model.train()
    for batch_index,(data,target) in enumerate(train_loader):
        #部署到device上
        data,target = data.to(device),target.to(device)
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
        if batch_index % 400 == 0:
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
            data,target = data.to(device),target.to(device)
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