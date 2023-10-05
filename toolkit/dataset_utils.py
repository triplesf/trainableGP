import os
import numpy as np
import random
import torch
from torchvision import datasets, transforms
from PIL import Image


def custom_grayscale_to_int(img):
    """
    Custom grayscale transformation that returns a grayscale image with pixel values as integers.

    Args:
        img (PIL.Image.Image): Input image.

    Returns:
        PIL.Image.Image: Grayscale image with pixel values as integers.
    """
    # Convert the input image to grayscale
    gray_img = img.convert('L')

    # Convert grayscale values to a NumPy array of integers
    gray_array = np.array(gray_img, dtype=np.uint8)

    return gray_array


def save_dataset_info(dataset, output_dir, dataset_name, dataset_type, dataset_size):
    """
    Save dataset information (name, size, class names) to a txt file.

    """

    num_classes = len(dataset.classes)
    image_size = dataset[0][0].shape[-2:]  # Get the size of the first image

    # Create a directory to save the .txt files
    os.makedirs(output_dir, exist_ok=True)

    # Save dataset information to a txt file
    with open(os.path.join(output_dir, f'{dataset_name}_{dataset_type}_info.txt'), 'w') as f:
        f.write(f'Dataset Name: {dataset_name}\n')
        f.write(f'Dataset Size: {dataset_size}\n')
        f.write(f'Number of Classes: {num_classes}\n')
        f.write(f'Image Width: {image_size[0]}\n')
        f.write(f'Image Height: {image_size[1]}\n')
        f.write('Class Names:\n')
        for i, class_name in enumerate(dataset.classes):
            f.write(f'{i + 1}. {class_name}\n')


def create_cifar10_train_samples(root, samples_per_class, output_dir):
    """
    Extracts a fixed number of samples from each class of the CIFAR-10 dataset,
    converts them to grayscale, and saves image data and labels as .npy files.

    Args:
        root (str): Root directory of the CIFAR-10 dataset.
        samples_per_class (int): Number of samples to select per class.
        output_dir (str): Directory to save the .npy files.
    """
    # Set a random seed for reproducible results
    random.seed(42)

    # Data transformations (convert to grayscale)
    transform = transforms.Lambda(custom_grayscale_to_int)

    # Load the training dataset using datasets.CIFAR10
    train_dataset = datasets.CIFAR10(root=root, train=True, transform=transform, download=False)

    # Get the number of classes in the CIFAR-10 dataset
    num_classes = len(train_dataset.classes)

    # Create a directory to save the .npy files
    os.makedirs(output_dir, exist_ok=True)

    # Create arrays to store image data and labels
    image_data = []
    labels = []

    # Create an empty dictionary to store indices for each class
    class_indices = {}

    # Initialize lists of indices for each class
    for i in range(num_classes):
        class_indices[i] = []

    # Iterate through the training dataset and store indices by class
    for i, (image, label) in enumerate(train_dataset):
        class_indices[label].append(i)

    # Create a new directory to save the .npy files
    for label, indices in class_indices.items():
        selected_indices = random.sample(indices, samples_per_class)
        for i in selected_indices:
            image_data.append(train_dataset[i][0])
            labels.append(label)

    # Save image data and labels as .npy files
    dataset_name = f'cifar10_{samples_per_class}'

    np.save(os.path.join(output_dir, f'{dataset_name}_train_data.npy'), np.array(image_data))
    np.save(os.path.join(output_dir, f'{dataset_name}_train_label.npy'), np.array(labels))

    dataset_size = samples_per_class * num_classes

    save_dataset_info(train_dataset, output_dir, dataset_name, "train", dataset_size)


def create_cifar10_test(root, output_dir):
    """
    Converts CIFAR-10 test set images to grayscale, and saves image data and labels as .npy files.

    Args:
        root (str): Root directory of the CIFAR-10 dataset.
        output_dir (str): Directory to save the .npy files.
    """
    # Data transformation (convert to grayscale)
    transform = transforms.Lambda(custom_grayscale_to_int)

    # Load the CIFAR-10 test dataset using datasets.CIFAR10
    test_dataset = datasets.CIFAR10(root=root, train=False, transform=transform, download=False)

    # Create a directory to save the .npy files
    os.makedirs(output_dir, exist_ok=True)

    # Create arrays to store image data and labels
    image_data = []
    labels = []

    # Iterate through the test dataset and store image data and labels
    for image, label in test_dataset:
        image_data.append(image)
        labels.append(label)

    # Save image data and labels as .npy files
    np.save(os.path.join(output_dir, 'cifar10_test_data.npy'), np.array(image_data))
    np.save(os.path.join(output_dir, 'cifar10_test_label.npy'), np.array(labels))

    save_dataset_info(test_dataset, output_dir, "cifar10", "test", len(test_dataset))


# Example usage
root_directory = '../dataset/raw/cifar10'  # Root directory of the CIFAR-10 dataset
samples_per_class = 320     # Number of samples to select per class
output_directory = '../dataset/processed/cifar10'  # Directory to save .npy files

create_cifar10_test(root_directory, output_directory)
create_cifar10_train_samples(root_directory, samples_per_class, output_directory)
