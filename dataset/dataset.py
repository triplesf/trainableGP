import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import os
import random
from torchvision import datasets, transforms
import torchvision
from sklearn.model_selection import KFold
# torch.manual_seed(1)


class CustomDataset(Dataset):
    def __init__(self, image_file, label_file, train_split=True, train=True, transform=None):
        self.images = np.load(image_file)
        self.labels = np.load(label_file)
        self.transform = transform

        # 如果需要划分数据集为训练集和验证集
        if train_split:
            unique_labels = np.unique(self.labels)
            train_indices = []
            val_indices = []

            for label in unique_labels:
                label_indices = np.where(self.labels == label)[0]
                np.random.shuffle(label_indices)
                split_idx = int(len(label_indices) * 0.8)  # 80% 训练集，20% 验证集
                train_indices.extend(label_indices[:split_idx])
                val_indices.extend(label_indices[split_idx:])

            if train:
                self.images = self.images[train_indices]
                self.labels = self.labels[train_indices]
            else:
                self.images = self.images[val_indices]
                self.labels = self.labels[val_indices]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class DataLoaderManager:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        if "gray" in config.data_name:
            transform = transforms.Compose([
                transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        self.samples_per_class = config.samples_per_class
        self.batch_size = config.batch_size
        self.class_num = 0
        if config.dataset_source == "numpy":
            root_path = os.path.join(config.data_path, config.data_name)
            train_data_path = root_path + '_train_data.npy'
            train_label_path = root_path + '_train_label.npy'
            test_data_path = root_path + '_test_data.npy'
            test_label_path = root_path + '_test_label.npy'
            self.train_dataset = CustomDataset(image_file=train_data_path, label_file=train_label_path,
                                               train_split=False, transform=transform)
            self.test_dataset = CustomDataset(image_file=test_data_path, label_file=test_label_path, train_split=False,
                                              transform=transform)
            self.class_num = self.read_dataset_info_from_file()

        elif config.dataset_source == "torch":
            self.train_dataset = torchvision.datasets.CIFAR10(root=config.data_path, train=True, download=False,
                                                              transform=transform)
            self.test_dataset = torchvision.datasets.CIFAR10(root=config.data_path, train=False, download=False,
                                                             transform=transform)
            self.class_num = self.read_dataset_info_by_torch()

    def get_selected_indices(self):
        class_num = self.class_num
        samples_per_class = self.samples_per_class
        train_dataset = self.train_dataset
        # test_dataset = self.test_dataset
        class_indices = [[] for _ in range(self.class_num)]
        selected_indices = []

        while len(selected_indices) < samples_per_class * class_num:
            index = random.randint(0, len(train_dataset) - 1)
            _, label = train_dataset[index]
            if len(class_indices[label]) < samples_per_class and index not in selected_indices:
                selected_indices.append(index)
                class_indices[label].append(index)
        return class_indices, selected_indices

    def get_dataloader(self):
        train_dataset = self.train_dataset
        test_dataset = self.test_dataset
        train_indices = []
        valid_indices = []
        class_indices, selected_indices = self.get_selected_indices()
        for ci, cv in enumerate(class_indices):
            random.shuffle(cv)
            split_point = int(len(cv) * 0.8)

            train_indices.extend(cv[:split_point])
            valid_indices.extend(cv[split_point:])

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)
        train_full_sampler = torch.utils.data.sampler.SubsetRandomSampler(selected_indices)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, sampler=train_sampler)
        valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, sampler=valid_sampler)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
        train_full_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, sampler=train_full_sampler)

        return train_loader, valid_loader, test_loader, train_full_loader, self.class_num

    def read_dataset_info_by_torch(self):
        logger = self.logger
        config = self.config

        if "SVHN" not in config.data_name:
            num_classes = len(self.train_dataset.classes)
        else:
            num_classes = 10

        logger.info("Dataset Info:")
        logger.info(f"Dataset Name: {config.data_name}")
        logger.info(f"Train Dataset Size: {len(self.test_dataset)}")
        logger.info(f"Test Dataset Size: {self.samples_per_class * num_classes}")
        logger.info(f"Number of Classes: {num_classes}")

        logger.info(f"Image Size: {self.train_dataset[0][0].shape}")
        logger.info('===============================================================================================')
        return num_classes

    def read_dataset_info_from_file(self):
        # Initialize variables to store extracted information
        logger = self.logger
        dataset_name = ""
        train_dataset_size = 0
        test_dataset_size = 0
        num_classes = 0
        image_width = ""
        image_height = ""
        image_size = ""

        train_dataset_path = os.path.join(self.config.data_path, f"{self.config.data_name}_train_info.txt")
        test_dataset_path = os.path.join(self.config.data_path, f"{self.config.data_name}_test_info.txt")

        try:
            with open(train_dataset_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split(': ')
                    if len(parts) == 2:
                        key, value = parts[0], parts[1]
                        if key == "Dataset Name":
                            dataset_name = value
                        elif key == "Dataset Size":
                            train_dataset_size = int(value)
                        elif key == "Number of Classes":
                            num_classes = int(value)
                        elif key == "Image Width":
                            image_width = value
                        elif key == "Image Height":
                            image_height = value
                        elif key == "Image Size":
                            image_size = value
        except FileNotFoundError:
            logger.warning(f"File not found: {train_dataset_path}")

        try:
            with open(test_dataset_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split(': ')
                    if len(parts) == 2:
                        key, value = parts[0], parts[1]
                        if key == "Dataset Size":
                            test_dataset_size = int(value)
        except FileNotFoundError:
            logger.warning(f"File not found: {test_dataset_path}")

        # Print extracted information
        logger.info("Dataset Info:")
        logger.info(f"Dataset Name: {dataset_name}")
        logger.info(f"Original Train Dataset Size: {train_dataset_size}")
        logger.info(f"Train Dataset Size: {self.samples_per_class * num_classes}")
        logger.info(f"Test Dataset Size: {test_dataset_size}")
        logger.info(f"Number of Classes: {num_classes}")
        if image_height and image_width:
            logger.info(f"Image Size: {image_width} * {image_height}")
        elif image_size:
            logger.info(f"Image Size: {image_size}")
        logger.info('===============================================================================================')

        # Return the number of classes
        return num_classes

    def cross_validation_generator(self):
        kf = KFold(n_splits=5, shuffle=True)
        ori_train_dataset = self.train_dataset
        test_dataset = self.test_dataset
        train_indices = []
        _, selected_indices = self.get_selected_indices()
        # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

        for train_index, val_index in kf.split(selected_indices):
            train_dataset = Subset(ori_train_dataset, [i for i in train_index])
            val_dataset = Subset(ori_train_dataset, [i for i in val_index])
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16)
            yield train_loader, val_loader

