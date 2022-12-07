import torch
import torch.nn.functional as F
from torch.linalg import norm
from torch.nn.functional import log_softmax
from functorch import jacrev
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

from utils import get_data, project_subspace
from models import LeNet, LeNetPL


def predict(model, x):
    pred = model(x)
    return log_softmax(pred)


def get_eigen(x, n_eigs=None):
    if n_eigs:
        eigvals, eigvecs = torch.lobpcg(x, k=n_eigs)
        eigvals, eigvecs = eigvals.real, eigvecs.real
    else:
        # Columns are eigvecs, project on transpose
        eigvals, eigvecs = torch.linalg.eig(x)
        eigvals, eigvecs = eigvals.real, eigvecs.real
    return eigvals, eigvecs


def find_path(s, d, model, alpha=0.1, T=5000, use_g=False, save_path=False,
              n_eigs=None, num_classes=10, update_steps=1000):

    x_hist = []
    x_t = s

    plt.imshow(x_t.view(28, 28).detach().cpu().numpy(), cmap='gray')
    plt.show()
    plt.imshow(d.view(28, 28).detach().cpu().numpy(), cmap='gray')
    plt.show()

    with torch.no_grad(), trange(T, desc='Traversing') as pbar:
        for t in pbar:  # calculate jacobian of log_softmax wrt x_t
            j = jacrev(predict, argnums=1)(model.lenet, x_t.unsqueeze(0))
            j = j.squeeze()

            if use_g:
                g = j.view(num_classes, -1).T @ j.view(num_classes, -1)
                eigvals, eigvecs = get_eigen(g, n_eigs)

                v = project_subspace(d - x_t, eigvecs.T)
            else:
                j_orth, _ = torch.linalg.qr(j.view(j.shape[0], -1).T)

                v = project_subspace(d - x_t, j_orth.T)

            # update
            x_t = x_t + alpha * (v.view(x_t.shape) / norm(v))

            # book keeping
            if t % update_steps == 0:
                probs = F.softmax(model.lenet(x_t.unsqueeze(0)))
                idx = torch.argmax(probs)
                norm_diff = norm(d - x_t)
                pbar.set_postfix(norm_diff=norm_diff.item(),
                                    p_class=idx.item(),
                                    prob=probs[0][idx].item())

                if save_path:
                    x_hist.append(x_t.view(28, 28).detach().cpu().numpy())

    plt.imshow(x_t.view(28, 28).detach().cpu().numpy(), cmap='gray')
    plt.show()


def gt_data():
    tdata, vdata = get_data()
    s, _ = tdata[3]
    d, _ = tdata[0]

    return s, d


def main():
    checkpoint_path = '/home/aditya/workspace/diffusion_manifold/lightning_logs/version_0/checkpoints/lmnist-epoch=20-val_loss=0.0303-val_acc=0.9914.ckpt'

    # get gt data
    s, d = gt_data()

    # load model
    lenet = LeNet()
    model = LeNetPL(lenet)
    model.load_from_checkpoint(checkpoint_path, lenet=lenet)
    find_path(s, d, model, use_g=True, T=10000)


if __name__ == '__main__':
    main()
