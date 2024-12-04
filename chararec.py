import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

print(torch.cuda.is_available())
#torch.set_default_tensor_type('torch.cuda.FloatTensor')


class Net(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)

    def forward(self, x):  # x : 入力
        z1 = F.relu(self.fc1(x))
        z2 = F.relu(self.fc2(z1))
        y = self.fc3(z2)
        return y


def train_model(model, train_loader, criterion, optimizer, device):
    train_loss = 0.0  # trainの損失用の変数を定義
    num_train = 0  # 学習回数の記録用の変数を定義

    # モデルを学習モードに変換
    model.train()

    # データの分割数分繰り返す
    # バッチサイズ分のデータで1回パラメータを修正する
    for i, (images, labels) in enumerate(train_loader):
        # batch数をカウント
        num_train += len(labels)

        # images, labels をdeviceに移動
        images, labels = images.view(-1, 28*28).to(device), labels.to(device)

        # 勾配を初期化
        optimizer.zero_grad()

        # １推論(順伝播)
        outputs = model(images)

        # ２損失の算出
        loss = criterion(outputs, labels)

        # ３勾配計算
        loss.backward()

        # ４パラメータの更新
        optimizer.step()

        # lossを加算
        train_loss += loss.item()

    # lossの平均値を取る
    train_loss = train_loss / num_train

    return train_loss


def test_model(model, test_loader, criterion, optimizer, device):
    test_loss = 0.0
    num_test = 0

    # modelを評価モードに変更
    model.eval()

    with torch.no_grad():  # 勾配計算の無効化
        for i, (images, labels) in enumerate(test_loader):
            num_test += len(labels)

            # images, labels をdeviceに移動
            images, labels = images.view(-1, 28*28).to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

        # lossの平均値を取る
        test_loss = test_loss / num_test
    return test_loss


def learning(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    train_loss_list = []
    test_loss_list = []

    # epoch数分繰り返す
    for epoch in range(1, num_epochs + 1, 1):
        train_loss = train_model(model, train_loader, criterion, optimizer, device=device)
        test_loss = test_model(model, test_loader, criterion, optimizer, device=device)

        print("epoch : {}, train_loss : {:.5f}, test_loss : {:.5f}".format(epoch, train_loss, test_loss))

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

    return train_loss_list, test_loss_list


# 訓練データ
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=False)
# 検証データ
test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor(),
                                          download=False)

batch_size = 1024

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

input_size = 28 * 28
hidden1_size = 1024
hidden2_size = 512
output_size = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)

model = Net(input_size, hidden1_size, hidden2_size, output_size).to(device)
print(model)

# 損失関数  criterion：基準
# CrossEntropyLoss：交差エントロピー誤差関数
criterion = nn.CrossEntropyLoss()

# 最適化法の指定  optimizer：最適化
# SGD：確率的勾配降下法
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 1000
train_loss_list, test_loss_list = learning(model, train_loader, test_loader, criterion, optimizer, num_epochs, device=device)

plt.plot(range(len(train_loss_list)), train_loss_list, c='b', label='train loss')
plt.plot(range(len(test_loss_list)), test_loss_list, c='r', label='test loss')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid()
plt.show()

model_scripted = torch.jit.script(model)
model_scripted.save("model_heavy.pth")

plt.figure(figsize=(20, 10))
for i in range(10):
    image, label = test_dataset[i]
    image = image.view(-1, 28*28).to(device)

    # 推論
    prediction_label = torch.argmax(model(image))

    ax = plt.subplot(1, 10, i+1)

    plt.imshow(image.detach().to('cpu').numpy().reshape(28, 28), cmap='gray')
    ax.axis('off')
    ax.set_title('label : {}\n Prediction : {}'.format(label, prediction_label), fontsize=15)
plt.show()
