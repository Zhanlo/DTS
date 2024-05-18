import logging

import math
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torchvision import datasets
import numpy as np
import torch
import torchvision.transforms as tv_transforms
from PIL import Image
import torchvision.transforms as transforms

import os


from randaugment import RandAugment

rng = np.random.RandomState(seed=1)
mean, std = {}, {}

mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
mean['stl10'] = [x / 255 for x in [112.4, 109.1, 98.6]]

std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]
std['stl10'] = [x / 255 for x in [68.4, 66.6, 68.5]]


data_path = "../../../../data"


class SimpleDataset(Dataset):
    def __init__(self, dataset, mode='test', dataset_name = None):
        self.dataset = dataset
        self.mode = mode
        self.dataset_name = dataset_name
        self.crop_ratio = 0.875

    def __getitem__(self, index):
        image = self.dataset['images'][index]
        label = self.dataset['labels'][index]

        if self.dataset_name == "CIFAR100":
            if self.mode == "train":
                self.set_aug(3)
                data0 = self.transform(Image.fromarray(image, 'RGB'))
                self.set_aug(4)
                data_noaug = self.transform(Image.fromarray(image, 'RGB'))
                return data0, data_noaug, label, index
            elif self.mode == "test":
                self.set_aug(5)
                data = self.transform(Image.fromarray(image, 'RGB'))
                return data, label, index


    def set_aug(self, method):


        if method == 3:
            # strong augment for cifar100
            transform_weak = transforms.Compose([
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=int(32 * (1 - self.crop_ratio)),
                                      padding_mode='reflect'),

                transforms.RandomHorizontalFlip(),
                RandAugment(3, 5),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
            self.transform = transform_weak
        elif method == 4:
            # weak augment for cifar100
            transform_weak = transforms.Compose([
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=int(32 * (1 - self.crop_ratio)),
                                      padding_mode='reflect'),

                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
            self.transform = transform_weak

        elif method == 5:
            # no augment for cifar100
            transform_weak = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
            self.transform = transform_weak




    def __len__(self):
        return len(self.dataset['images'])


def split_l_u(train_set, n_labels, n_unlabels, tot_class=6, ratio = 0.5):
    images = train_set["images"]
    labels = train_set["labels"]
    classes = np.unique(labels)


    n_labels_per_cls = n_labels // tot_class

    n_unlabels_per_cls = int(n_unlabels*(1.0-ratio)) // tot_class



    if(tot_class < len(classes)):
        n_unlabels_shift = (n_unlabels - (n_unlabels_per_cls * tot_class)) // (len(classes) - tot_class)

    l_images = []
    l_labels = []
    u_images = []
    u_labels = []


    for c in classes[:tot_class]:

        cls_mask = (labels == c)

        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        l_images += [c_images[:n_labels_per_cls]]
        l_labels += [c_labels[:n_labels_per_cls]]

        u_images += [c_images[n_labels_per_cls:n_labels_per_cls+n_unlabels_per_cls]]
        u_labels += [c_labels[n_labels_per_cls:n_labels_per_cls+n_unlabels_per_cls]]


    for c in classes[tot_class:]:

        cls_mask = (labels == c)
        c_images = images[cls_mask]

        c_labels = labels[cls_mask]
        u_images += [c_images[:n_unlabels_shift]]
        u_labels += [c_labels[:n_unlabels_shift]]



    l_train_set = {"images": np.concatenate(l_images, 0), "labels": np.concatenate(l_labels, 0)}
    u_train_set = {"images": np.concatenate(u_images, 0), "labels": np.concatenate(u_labels, 0)}



    indices = rng.permutation(len(l_train_set["images"]))
    l_train_set["images"] = l_train_set["images"][indices]
    l_train_set["labels"] = l_train_set["labels"][indices]

    indices = rng.permutation(len(u_train_set["images"]))
    u_train_set["images"] = u_train_set["images"][indices]
    u_train_set["labels"] = u_train_set["labels"][indices]


    return l_train_set, u_train_set

def split_test(test_set, tot_class=6):
    images = test_set["images"]
    labels = test_set['labels']

    classes = np.unique(labels)
    l_images = []
    l_labels = []
    for c in classes[:tot_class]:

        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]


        l_images += [c_images[:]]
        l_labels += [c_labels[:]]

    test_set = {"images": np.concatenate(l_images, 0), "labels":np.concatenate(l_labels,0)}


    indices = rng.permutation(len(test_set["images"]))
    test_set["images"] = test_set["images"][indices]
    test_set["labels"] = test_set["labels"][indices]

    return test_set








def load_cifar100():
    splits = {}
    for train in [True, False]:
        dataset = datasets.CIFAR100(data_path, train, download=True)
        data = {}
        data['images'] = dataset.data
        data['labels'] = np.array(dataset.targets)
        splits["train" if train else "test"] = data
    return splits.values()





def get_dataloaders(dataset, n_labels, n_unlabels, n_valid, tot_class, ratio):

    rng = np.random.RandomState(seed=1)
    dataset_name = None

    if dataset == "CIFAR100":
        dataset_name = "CIFAR100"
        train_set, test_set = load_cifar100()



    #permute index of training set
    indices = rng.permutation(len(train_set['images']))

    train_set['images'] = train_set['images'][indices]
    train_set['labels'] = train_set['labels'][indices]

    #split training set into training and validation
    train_images = train_set['images'][n_valid:]
    train_labels = train_set['labels'][n_valid:]
    validation_images = train_set['images'][:n_valid]
    validation_labels = train_set['labels'][:n_valid]

    validation_set = {'images': validation_images, 'labels': validation_labels}
    train_set = {'images': train_images, 'labels': train_labels}


    validation_set = split_test(validation_set, tot_class=tot_class)
    test_set = split_test(test_set, tot_class=tot_class)

    l_train_set, u_train_set = split_l_u(train_set, n_labels, n_unlabels, tot_class=tot_class, ratio=ratio)

    logging.info("Unlabeled data in distribuiton : {}, Unlabeled data out distribution : {}".format(
          np.sum(u_train_set['labels'] < tot_class), np.sum(u_train_set['labels'] >= tot_class)))



    l_train_set = SimpleDataset(l_train_set, mode='train', dataset_name=dataset_name)
    u_train_set = SimpleDataset(u_train_set, mode='train', dataset_name=dataset_name)
    validation_set = SimpleDataset(validation_set, mode='test', dataset_name=dataset_name)
    test_set = SimpleDataset(test_set, mode='test', dataset_name=dataset_name)


    logging.info("labeled data : {}, unlabeled data : {},  training data : {}".format(
        len(l_train_set), len(u_train_set), len(l_train_set) + len(u_train_set)))
    logging.info("validation data : {}, test data : {}".format(len(validation_set), len(test_set)))


    return l_train_set, u_train_set, validation_set, test_set






















