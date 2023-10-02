import torch
import numpy as np
import skimage
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

torch.manual_seed(1)  # reproducible

transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
])


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.label = label
        self.transforms = transform  # 转为tensor形式
        if data.ndim == 3:
            self.data = data
        elif data.ndim == 4 and data.shape[1] == 1:
            self.data = np.squeeze(data, axis=1)
        else:
            raise ValueError("Input array must be three-dimensional or four-dimensional "
                             "with the second dimension equal to 1.")

    def __getitem__(self, index):
        hdct= self.data[index, :, :]  # 读取每一个npy的数据
        hdct = Image.fromarray(np.uint8(hdct)) #转成image的形式
        hdct = self.transforms(hdct)  #转为tensor形式
        label = self.label[index]
        return hdct, label  #返回数据还有标签

    def __len__(self):
        return self.data.shape[0]  # 返回数据的总个数


def prepare_dataset(data_root, data_name):
    train_dataset_path = os.path.join(data_root, data_name)
    test_dataset_path = os.path.join(data_root, data_name.split('_')[0])

    x_train = np.load(train_dataset_path+'_train_data.npy')
    # 60 * 40尺寸 150张图片  shape = (150, 60, 40) 人脸二维分类图
    y_train = np.load(train_dataset_path+'_train_label.npy')
    # 标签 两类 0和1 shape=150
    x_test = np.load(test_dataset_path+'_test_data.npy')
    y_test = np.load(test_dataset_path+'_test_label.npy')
    train_num = len(x_train)

    data_size = x_train.shape[0] # 数据集个数
    arr = np.arange(data_size) # 生成0到datasize个数
    np.random.shuffle(arr) # 随机打乱arr数组
    x_train = x_train[arr] # 将data以arr索引重新组合
    y_train = y_train[arr] # 将label以arr索引重新组合

    slice_num = int(train_num / 5)
    train_data = MyDataset(x_train[slice_num:], y_train[slice_num:])
    val_data = MyDataset(x_train[:slice_num], y_train[:slice_num])
    test_data = MyDataset(x_test, y_test)
    all_train_data = MyDataset(x_train, y_train)
    return train_data, val_data, test_data, all_train_data


def prepare_data(data_root, dataSetName):
    # dataSetName = 'f1_ours'
    dataSetPath = os.path.join(data_root, dataSetName)
    x_train = np.load(dataSetPath+'_train_data.npy')
    # 60 * 40尺寸 150张图片  shape = (150, 60, 40) 人脸二维分类图
    y_train = np.load(dataSetPath+'_train_label.npy')
    # 标签 两类 0和1 shape=150
    x_test = np.load(dataSetPath+'_test_data.npy')
    y_test = np.load(dataSetPath+'_test_label.npy')
    return x_train, y_train, x_test, y_test


def read_dataset_info(data_path, data_name, logger):
    # Initialize variables to store extracted information
    dataset_name = ""
    train_dataset_size = 0
    test_dataset_size = 0
    num_classes = 0
    image_width = ""
    image_height = ""

    train_dataset_path = os.path.join(data_path, f"{data_name}_train_info.txt")
    test_dataset_path = os.path.join(data_path, f"{data_name.split('_')[0]}_test_info.txt")

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
    logger.info(f"Train Dataset Size: {train_dataset_size}")
    logger.info(f"Test Dataset Size: {test_dataset_size}")
    logger.info(f"Number of Classes: {num_classes}")
    logger.info(f"Image Size: {image_width} * {image_height}")
    logger.info('===============================================================================================')

    # Return the number of classes
    return num_classes

