import torch
import torchvision
import torchvision.transforms as transforms
import os
import dataset.cifar10
from misc import Cutout
import torchvision.transforms as T
def apply_transform(images,img_augs):
    MEAN, STD = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    t=  transforms.Compose([
        T.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        img_augs[0],img_augs[1],
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize(MEAN, STD),
            ])
    return torch.stack([t(im) for im in images])
    
def data_loader_func_no_train_transform(batch_sizes, path_to_dataset):
    """
    Preparing the transform, dataset, dataloader, etc.
    :param batch_sizes:
    :param epochs:
    :return: none
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])
    transform_tensor = transforms.Compose([transforms.ToTensor()])
    if not os.path.exists(path_to_dataset):
        os.makedirs(path_to_dataset)

    # trainset = torchvision.datasets.CIFAR10(root=path_to_dataset, train=True,
    #                                         download=True, transform=transform_train)
    #remove aut
    teacher_train_set = dataset.cifar10.CIFAR10(root=path_to_dataset, transform=transform_tensor, split='teacher_train')
    teacher_train_loader = torch.utils.data.DataLoader(teacher_train_set, batch_size=batch_sizes,
                                                       shuffle=True, num_workers=0)

    dev_set = dataset.cifar10.CIFAR10(root=path_to_dataset, transform=transform_test, split='dev')
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=batch_sizes,
                                             shuffle=False, num_workers=0)

    student_train_set = dataset.cifar10.CIFAR10(root=path_to_dataset, transform=transform_tensor, split='student_train')
    student_train_loader = torch.utils.data.DataLoader(student_train_set, batch_size=batch_sizes,
                                                       shuffle=True, num_workers=0)

    test_set = dataset.cifar10.CIFAR10(root=path_to_dataset, transform=transform_test, split='test')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_sizes,
                                              shuffle=False, num_workers=0)

    classes = ('airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    dataloader = {'teacher_train_loader': teacher_train_loader,
                  'dev_loader': dev_loader,
                  'student_train_loader': student_train_loader,
                  'test_loader': test_loader,
                  'classes': classes}

    return dataloader


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#
#     # functions to show an image
#     def imshow_2(img):
#         img = img / 2 + 0.5  # unnormalize
#         npimg = img.numpy()
#         plt.imshow(np.transpose(npimg, (1, 2, 0)))
#         plt.show()
#
#
#     def imshow(inp, title=None):
#         """Imshow for Tensor."""
#         inp = inp.numpy().transpose((1, 2, 0))
#         mean = np.array((0.4914, 0.4822, 0.4465))
#         std = np.array((0.2023, 0.1994, 0.2010))
#         inp = std * inp + mean
#         inp = np.clip(inp, 0, 1)
#         plt.imshow(inp)
#         plt.show()
#         if title is not None:
#             plt.title(title)
#         plt.pause(0.001)  # pause a bit so that plots are updated
#
#
#     # get some random training images
#     dataloader = data_loader_func(batch_sizes=50, path_to_dataset='../data')
#     dataiter = iter(dataloader['test_loader'])
#     images, labels = dataiter.next()
#     print(' '.join('%5s' % dataloader['classes'][labels[j]] for j in range(50)))
#     # show images
#     imshow(torchvision.utils.make_grid(images))
    # print labels

    # class_img_list = [0]*10
    # total = 0
    # for images, labels in trainloader:
    #     for label in labels:
    #         class_img_list[label.item()]+=1
    #     total+=labels.size(0)
    #
    # print(class_img_list)
    # print(total)
    #
    # class_img_list = [0] * 10
    # total = 0
    # for images, labels in testloader:
    #     for label in labels:
    #         class_img_list[label.item()] += 1
    #     total += labels.size(0)
    #
    # print(class_img_list)
    # print(total)
