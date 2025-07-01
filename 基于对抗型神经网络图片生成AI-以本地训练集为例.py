import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
 
# 生成器网络
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, img_dim),
            nn.Tanh()
        )
 
    def forward(self, x):
        return self.model(x)
 
 
# 判别器网络
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        return self.model(x)
 
# 获取图像
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = os.listdir(root_dir)
 
    def __len__(self):
        return len(self.image_list)
 
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(img_name).convert('RGB')
 
        if self.transform:
            image = self.transform(image)
 
        return image
 
device = torch.device("cuda")  # 使用GPU进行训练
img_dim = 28 * 28   # 输入图像的维度
noise_dim = 100  # 噪声向量的维度
batch_size = 128
lr = 0.00001
root_dir = "E:\\程序\\pythonProject\\train_image"
# 加载本地数据集作为训练集
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset =CustomDataset(root_dir, transform=transform)#获取图像
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#加载图像
 
generator = Generator().to(device)
discriminator = Discriminator().to(device)
 
criterion = nn.BCELoss()  # 二分类交叉熵损失函数
 
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
 
num_epochs = 10000  #训练的次数
print(train_loader)
x = []
y1 = []
y2 = []
for epoch in range(num_epochs):
    for i, (real_images) in enumerate(train_loader):
        real_images = real_images.view(-1, img_dim).to(device)
        batch_size = real_images.shape[0]
 
        # 训练判别器
        optimizer_D.zero_grad()
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
 
        # 真实图像的判别器损失
        outputs = discriminator(real_images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
 
        # 生成器生成的图像的判别器损失
        z = torch.randn(batch_size, noise_dim).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
 
        # 计算判别器的总损失并进行反向传播和优化
        d_loss = d_loss_real + d_loss_fake
        optimizer_D.step()
 
        # 训练生成器
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, noise_dim).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images)
 
        # 生成器生成的图像的判别器损失（我们希望生成的图像能够被判别为真实）
        g_loss = criterion(outputs, real_labels)
 
        # 反向传播和优化
        optimizer_G.step()
 
    # 每个epoch结束后输出损失和生成的图像
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, D(x): {real_score.mean().item():.2f}, D(G(z)): {fake_score.mean().item():.2f}")
    x.append(str(epoch + 1))
    y1.append(float(d_loss))
    y2.append(float(g_loss))
#生成g_loss与d_loss图表
plt.figure()
plt.title('loss during training')  #标题
plt.plot(x, y1, label="d_loss")
plt.plot(x,y2, label="g_loss")
plt.legend()
plt.grid()
plt.show()
#生成生成器生成出来的马赛克.dog
fake_images2 = fake_images.cpu().clone()
toPIL = transforms.ToPILImage()
img_PIL = toPIL(fake_images2)
img_PIL.show()