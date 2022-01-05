import os

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def data_loader(img_size, train_batch_size, test_batch_size, dt_type='cifar10'):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if dt_type == "cifar10":
        train_set = datasets.CIFAR10(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        test_set = datasets.CIFAR10(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test)

    else:
        train_set = datasets.CIFAR100(root="./data",
                                      train=True,
                                      download=True,
                                      transform=transform_train)
        test_set = datasets.CIFAR100(root="./data",
                                     train=False,
                                     download=True,
                                     transform=transform_test)

    train_sampler = RandomSampler(train_set)
    test_sampler = SequentialSampler(test_set)
    train_loader = DataLoader(train_set,
                              sampler=train_sampler,
                              batch_size=train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(test_set,
                             sampler=test_sampler,
                             batch_size=test_batch_size,
                             num_workers=4,
                             pin_memory=True) if test_set is not None else None

    return train_loader, test_loader


def find_newest_model(models_dir):
    model_file_lists = os.listdir(models_dir)
    model_file_lists.sort(key=lambda fn: os.path.getmtime(models_dir + "/" + fn)
    if not os.path.isdir(models_dir + "/" + fn) else 0)

    model_name = model_file_lists[-1]
    model_file = os.path.join(models_dir, model_file_lists[-1])
    if model_name.find(".bin") == -1:
        return None, None

    return model_name, model_file