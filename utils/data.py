import torch
import torchvision
import torchvision.transforms as transforms


def get_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                          download=True, transform=transform)

    return trainset, testset


def get_data2(args):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(64),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0.1307, 0.3081),
        ]
    )

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                          download=True, transform=transform)

    return trainset, testset


def get_data_diff():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(64)
         ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                          download=True, transform=transform)

    return trainset, testset


def prep_img_for_classifier(img):
    transform = transforms.Compose(
        [transforms.Resize(28),
         transforms.Normalize((0.1307,), (0.3081,))])

    return transform(img)



def get_loaders(batch_size=512, num_workers=4):
    trainset, testset = get_data()

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)
    return trainloader, testloader


def get_loaders2(batch_size=512, num_workers=4):
    trainset, testset = get_data2()

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)
    return trainloader, testloader
