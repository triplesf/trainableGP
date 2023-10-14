import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import random


# 定义四层卷积神经网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16384, 10)
        # self.fc2 = nn.Linear(1000, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        # x = self.dropout(x)
        # x = F.relu(self.fc1(x))
        x = self.fc1(x)

        return x


def main():
    samples_per_class = 20
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    root_path = "/home/triplef/code/trainableGP/dataset/raw/cifar10"
    trainset = torchvision.datasets.CIFAR10(root=root_path, train=True, download=False, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=root_path, train=False, download=False, transform=transform)

    # Randomly select 40 images from each class in the training set
    class_indices = [[] for _ in range(10)]
    selected_indices = []

    while len(selected_indices) < samples_per_class * 10:
        index = random.randint(0, len(trainset) - 1)
        _, label = trainset[index]
        if len(class_indices[label]) < samples_per_class and index not in selected_indices:
            selected_indices.append(index)
            class_indices[label].append(index)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(selected_indices))
    test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
    # 模型、优化器和损失函数
    net = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.00001)

    # 训练
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    for epoch in range(400):  # 训练100个epoch
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            # print(f"Batch {i + 1}: Input shape {inputs.shape}, Label shape {labels.shape}")

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

    print("Finished Training")

    # 在测试集上评估模型
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on test dataset: {100 * correct / total:.2f}%")


if __name__ == '__main__':
    main()
