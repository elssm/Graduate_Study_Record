import torch
import torchvision
import torch.nn as nn
from torchvision import datasets,transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.utils.data import DataLoader

#还原真实数据
def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0,1) #将随机变化的数值限制在一个给定的区间
    out = out.view(-1,1,28,28)
    return out

batch_size = 64
epochs = 50
z_dimension = 100

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

train_set = datasets.MNIST("data",train=True,transform=transforms,download=True)
train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.dis(x)
        x = x.squeeze(1)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(100,256), #输入一个100维的0～1之间的高斯分布
            nn.ReLU(True),
            nn.Linear(256,256),
            nn.ReLU(True),
            nn.Linear(256,784),
            nn.Tanh()
        )
    def forward(self,x):
        x = self.gen(x)
        x = x.squeeze(-1)
        return x

Dis = Discriminator()
Gen = Generator()

criterion = nn.BCELoss()
Dis_optimizer = torch.optim.Adam(Dis.parameters(),lr=0.0003)
Gen_optimizer = torch.optim.Adam(Gen.parameters(),lr=0.0003)

for epoch in range(epochs):
    for batch_idx, (img,_) in enumerate(train_loader):
        num_img = img.size(0) #获取图片大小 64
        # print(num_img)
        #训练判别器
        img = img.view(num_img,-1) #拉平成64*784
        real_img = Variable(img) #将tensor变成Variable计算
        real_label = Variable(torch.ones(num_img)) #真图片label为1
        fake_label = Variable(torch.zeros(num_img)) #假图片label为0

        #计算真实图片的loss
        real_out = Dis(real_img) #图片放入判别器
        d_loss_real = criterion(real_out,real_label) #计算loss
        real_scores = real_out

        #计算假图片的loss
        z = Variable(torch.randn(num_img,z_dimension)) #随机生成一些噪声
        fake_img = Gen(z)
        fake_out = Dis(fake_img)
        d_loss_fake = criterion(fake_out,fake_label)
        fake_scores = fake_out

        d_loss = d_loss_real + d_loss_fake
        Dis_optimizer.zero_grad()
        d_loss.backward()
        Dis_optimizer.step()


        #训练生成器
        z = Variable(torch.randn(num_img, z_dimension))  # 随机生成一些噪声
        fake_img = Gen(z)
        output = Dis(fake_img)
        g_loss = criterion(output,real_label)

        Gen_optimizer.zero_grad()
        g_loss.backward()
        Gen_optimizer.step()

        if batch_idx % 100 ==0:
            print('Epoch {},d_loss: {:.6f},g_loss: {:.6f},D real: {:.6f},D fake: {:.6f}'.format(
                epoch,d_loss.item(),g_loss.item(),real_scores.data.mean(),fake_scores.data.mean()
            ))
        if epoch == 0:
            # print(real_img.shape)
            real_images = to_img(real_img.data)
            # print(real_images.shape)
            save_image(real_images.data,'img/real_images.png')
        fake_images = to_img(fake_img.data)
        save_image(fake_images.data,"img/fake_images-{}.png".format(epoch))

