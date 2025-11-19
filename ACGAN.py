import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.autograd as autograd
#from W_distance import WassersteinLoss

# 数据加载和预处理部分保持不变
df = pd.read_csv('E:/PAPER_WRITE/paper/practice1/code/data/data_ACGAN.csv')

features = df.iloc[:, :14].values.astype(np.float32)
labels = df.iloc[:, 14].values
print(df)
min_val = np.min(features, axis=0)
max_val = np.max(features, axis=0)
features = 2.0 * (features - min_val) / (max_val - min_val) - 1.0

features_tensor = torch.tensor(features)
labels_tensor = torch.tensor(labels)

batch_size = 25
latent_dim = 100
num_classes = len(np.unique(labels))
print(num_classes)
num_epochs = 20000

dataset = TensorDataset(features_tensor, labels_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 定义ACWGAN-GP的生成器和判别器
class Generator(nn.Module):
    def __init__(self, latent_dim, num_features, num_classes):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        self.model = nn.Sequential(
            nn.Linear(latent_dim * 2, 50),
            nn.LeakyReLU(0.2),
            nn.Linear(50, 45),#50/40
            nn.LeakyReLU(0.2),
            nn.Linear(45, 30),
            nn.LeakyReLU(0.2),
            nn.Linear(30, 14),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        combined_input = torch.cat([noise, label_input], dim=1)
        return self.model(combined_input)

class Discriminator(nn.Module):
    def __init__(self, features_num, label_dim):
        super(Discriminator, self).__init__()
        self.features_num = features_num
        self.label_dim = label_dim
        self.model = nn.Sequential(
            nn.Linear(features_num, 40),
            nn.LeakyReLU(0.2),
            nn.Linear(40, 20),
            nn.LeakyReLU(0.2),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )
        self.auxiliar = nn.Sequential(
            nn.Linear(features_num, 10),
            nn.LeakyReLU(0.2),
            nn.Linear(10, 3),
            #nn.Softmax(dim=1)
        )

    def forward(self, x):
        validity = self.model(x)
        #print('validity=', validity)
        label = self.auxiliar(x)
        return validity, label



# 初始化模型
generator = Generator(latent_dim, num_features=14, num_classes=num_classes)
discriminator = Discriminator(features_num=14, label_dim=num_classes)

# 定义损失函数和优化器
def compute_gradient_penalty(D, real_samples, fake_samples):
    """计算梯度惩罚"""
    alpha = torch.rand(real_samples.size(0), 1)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _ = D(interpolates)
    fake = torch.ones(real_samples.size(0), 1)

    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


adversarial_loss = nn.BCELoss()
aux_loss = nn.CrossEntropyLoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0015, betas=(0.90, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.003, betas=(0.7, 0.999))
length_x = 0
d_loss_list = []
g_loss_list = []
mean_mses = []
if __name__ == "__main__":
    # 训练循环
    for epoch in range(num_epochs):
        mses = []

        for i, (real_features, real_labels) in enumerate(dataloader):
            batch_size = real_features.size(0)

            # 真实数据的标签
            valid = torch.ones(batch_size, 1)
            fake = torch.zeros(batch_size, 1)

            # ---------------------
            #  训练判别器
            # ---------------------
            optimizer_D.zero_grad()

            # 真实数据的损失
            real_validity, real_pred = discriminator(real_features)
            d_real_loss = adversarial_loss(real_validity, valid) + aux_loss(real_pred, real_labels)

            # 生成假数据
            noise = torch.randn(batch_size, latent_dim)
            fake_labels = torch.randint(0, num_classes, (batch_size,))
            fake_features = generator(noise, real_labels)

            # 假数据的损失
            fake_validity, fake_pred = discriminator(fake_features.detach())
            if i == 1 and epoch==1:
                print('fake_validity', fake_validity)
            d_fake_loss = adversarial_loss(fake_validity, fake)

            # 梯度惩罚
            gradient_penalty = compute_gradient_penalty(discriminator, real_features, fake_features)

            d_loss = d_real_loss + d_fake_loss + 0.01 * gradient_penalty

            d_loss_list.append(d_loss.item())
            d_loss.backward(retain_graph=True)  # 设置retain_graph=True
            optimizer_D.step()

            # ---------------------
            #  训练生成器
            # ---------------------
            optimizer_G.zero_grad()

            # 生成假数据并计算损失
            fake_valid, fake_label_pr = discriminator(fake_features)
            g_fake_loss = adversarial_loss(fake_valid, valid)
            g_class_loss = aux_loss(fake_label_pr, real_labels)
            g_loss = g_class_loss + g_fake_loss
            g_loss_list.append(g_loss.item())

            g_loss.backward(retain_graph=True)  # 设置retain_graph=True
            optimizer_G.step()

            #停止条件
            # 计算均方误差
            tensor_flat = fake_validity.clone()
            mse = torch.mean((tensor_flat - 0.5) ** 2)
            mses.append(mse.item())
            if i==len(dataloader)-1:
                print(mses, np.mean(mses))
                mean_mses.append(np.mean(mses))
            #print('i', i)
            #print(mse.size)
            #print("均方误差：", mse.item())
            '''
            if mse.item() < 1e-4:
                break
             '''
        # 打印训练进度
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")
            print('d_real_loss', d_real_loss, 'd_fake_loss', d_fake_loss, 'gradient_penalty', gradient_penalty)

        if epoch>5000 and np.mean(mses)<0.04:
            print('epoch', epoch)
            length_x = epoch
            break

    plt.scatter([x for x in range(len(mean_mses))], mean_mses)
    plt.show()
    print('len(mean_mses)', len(mean_mses))
    # 生成示例数据
    with torch.no_grad():
        noise = torch.randn(400, latent_dim)
        labels = torch.randint(2, num_classes, (400,))
        print(labels)
        generated_features = generator(noise, labels).numpy()

    # 反归一化生成的特征
    generated_features = (generated_features + 1) / 2 * (max_val - min_val) + min_val

    print("Generated Features (示例):")
    print(generated_features)

    # 绘图
    plt.plot(d_loss_list, label='Discriminator Loss')
    plt.plot(g_loss_list, label='Generator Loss')
    plt.legend()
    plt.show()

    loss_df = pd.DataFrame()
    loss_df['Discriminator Loss'] = d_loss_list
    loss_df['Generator Loss'] = g_loss_list
    loss_df.to_csv('E:/PAPER_WRITE/paper/practice1/code/data/loss_df.csv', index=False)


    '''
    'Vec_std','X_mean','X_std/mean*100','atomic_radius_mean','atomic_radius_std/mean*100','electron_affinity_std/mean*100','group_std/mean*100','min_oxidation_state_std/mean*100','mendeleev_no_mean','atomic_radius_calculated_std','Processing_number','Cu','oumiga'
    '''
    generated_features_df = pd.DataFrame(generated_features, columns=['Processing_number', 'Ag', 'Al', 'Cr', 'Cu', 'Fe',
                                                                      'Mg', 'Mn', 'Ni', 'Sc', 'Si', 'Ti', 'Zn', 'Zr'])

    generated_features_df.iloc[:, 0] = generated_features_df.iloc[:, 0].round(10)
    generated_features_df.round(10).to_excel('E:/PAPER_WRITE/paper/practice1/code/data/new_samples.xlsx', index=False)

    # 保存生成器
    torch.save(generator.state_dict(), 'E:/PAPER_WRITE/paper/practice1/code/data/generator.pth')
    print("生成器已保存为 'generator.pt")
