import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image

#加载数据
train_data = datasets.MNIST("data",train=True,transform=transforms.ToTensor())
print(train_data.data.size())   #训练数据集的大小 60000*28*28
print(train_data.targets.size()) #标签个数：60000个
# plt.imshow(train_data.data[1].numpy())
# plt.show()
train_loader = DataLoader(train_data,batch_size=100,shuffle=True)

#定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,stride=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2),
            nn.Conv2d(16,8,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=1)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8,16,kernel_size=3,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16,8,kernel_size=5,stride=3,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8,1,kernel_size=2,stride=2,padding=1)
        )

    def forward(self,x):
        encoder = self.encoder(x)
        decode = self.decoder(encoder)
        return encoder,decode

#定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,32,5,padding=2), #32*28*28
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2,stride=2), #32*14*14
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,5,padding=2), #64*14*14
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2,stride=2) #64*7*7
        )

        self.fc = nn.Sequential(
            nn.Linear(64*7*7,1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x


#定义参数
gen = Generator()
dis = Discriminator()
loss_func = nn.BCELoss()
loss_func_auto = nn.MSELoss()
g_optimizer = torch.optim.Adam(gen.parameters(),lr=0.0003)
d_optimizer = torch.optim.Adam(dis.parameters(),lr=0.0002)

#训练
for epoch in range(10):
    D_loss = 0
    G_loss = 0
    for batch_idx,(data,target) in enumerate(train_loader):
        size = data.shape[0]
        real_img = Variable(data)

        #判别器训练
        real_label = Variable(torch.ones(size,1))
        false_label = Variable(torch.zeros(size,1))
        real_out = dis(real_img)
        d_loss_real = loss_func(real_out,real_label)

        encoder,false_img = gen(data)
        false_img = Variable(false_img)
        false_out = dis(false_img)
        d_loss_false = loss_func(false_out,false_label)

        d_loss = d_loss_real+d_loss_false
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        D_loss+=d_loss.item()
        # if batch_idx % 10 ==0:
        print("Epoch",epoch,'|d_loss: %.4f' % d_loss.data.numpy())

        #生成器训练
        encoded,decoded = gen(real_img)
        output = dis(decoded)
        g_loss = loss_func(output,real_label)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        G_loss+=g_loss.item()
        # print(false_img.shape)
        if batch_idx %100 ==0:
            print("Epoch: ",epoch,"|d_loss: %.4f" % d_loss.data.numpy(),'|g_loss: %.4f' % g_loss.data.numpy())
            save_image(false_img.data[:25], "images/%d.png" % batch_idx, nrow=5, normalize=True)
    print('epoch: {},D_Loss: {:.6f}, G_Loss: {:.6f}'.format(epoch,D_loss/len(train_loader),G_loss/len(train_loader)))

