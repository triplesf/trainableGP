import torch
import torch.nn as nn
import torch.optim as optim
from Resnet20.resnet20 import ResNet20
from low_layer_cnn.low_layer import SimpleCNN
import torchvision.transforms as transforms
import torchvision
import random
import numpy as np

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


# 训练模型

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

samples_per_class = 10

transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
])

root_path = "/home/triplef/code/trainableGP/dataset/raw/FMNIST"

trainset = torchvision.datasets.FashionMNIST(root=root_path, train=True, download=False, transform=transform)
testset = torchvision.datasets.FashionMNIST(root=root_path, train=False, download=False, transform=transform)
result = []

for exp_round in range(10):
    print(f"Start {exp_round} round")

    net = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    # 训练

    net.to(device)

    # Randomly select X images from each class in the training set
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

    for epoch in range(400):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

    # 在测试集上评估模型
    net.eval()
    correct = 0.0
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

    result.append(100 * correct / total)

    print(f"Finish {exp_round} round\n")

for ei, r in enumerate(result):
    print("Epoch {} Result: {}".format(ei, r))
result_np = np.array(result)
print("Mean: {}".format(np.mean(result_np)))
print("Std: {}".format(np.std(result_np)))
