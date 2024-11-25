import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")


def create_data_loaders(dataset_path, batch_size, img_size):
    # Training data transform with augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    # Validation and test transform without augmentation
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    # Create datasets for each split
    train_dataset = datasets.ImageFolder(
        os.path.join(dataset_path, 'train'),
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(dataset_path, 'validation'),
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        os.path.join(dataset_path, 'test'),
        transform=eval_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader