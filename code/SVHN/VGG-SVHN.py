import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.utils.model_zoo as model_zoo

import torch.nn as nn
import torch

batch_size = 16
epochs = 50
transforms = transforms.Compose([
    transforms.Resize((224 , 224)), #(64,128)(128,128)
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

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512*7*7, 2048), #512*7*7
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg(model_name="vgg11", **kwargs):
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)
    model = VGG(make_features(cfg), **kwargs)
    return model

model_name = "vgg11"
model = vgg(model_name=model_name, num_classes=10, init_weights=True)

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