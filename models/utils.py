import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def plot_images(images):
    # plt.figure(figsize=(32, 32))
    # plt.imshow(torch.cat([
    #     torch.cat([i for i in images.cpu()], dim=-1),
    # ], dim=-2).permute(1, 2, 0).cpu())
    grid = torchvision.utils.make_grid(images)
    plt.imshow(grid.cpu().permute(1, 2, 0))
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    plt.imshow(grid.cpu().permute(1, 2, 0))
    plt.savefig(path)


def get_data(args):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(64),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0.1307, 0.3081),
        ]
    )
    batch_size = args.batch_size
    dataset = torchvision.datasets.MNIST("./mnist/", train=True, transform=transforms, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
