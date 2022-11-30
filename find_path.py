import torch
import torch.nn.functional as F
from torch.linalg import norm
from torch.nn.functional import log_softmax
from functorch import jacrev
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from utils import get_data, project_subspace
from models import LeNet, LeNetPL


def predict(model, x):
    pred = model(x)
    return log_softmax(pred)


def find_path(s, d, model, alpha=0.1, T=10000, save_path=False):

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

            # orthonomalize for projection
            # j = gram_schmidt_orthogonalize(j)

            # TODO check need for 10 * -1 shapes, does 10* 28* 28 make a difference?
            # orthogonalize
            j_orth, _ = torch.linalg.qr(j.view(j.shape[0], -1).T)

            # project d-x_t onto j
            v = project_subspace(d - x_t, j_orth.T)

            # update
            x_t = x_t + alpha * (v.view(x_t.shape) / norm(v))

            # book keeping
            if t % 1000 == 0:
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


def main():
    checkpoint_path = '/home/aditya/workspace/diffusion_manifold/lightning_logs/version_0/checkpoints/lmnist-epoch=20-val_loss=0.0303-val_acc=0.9914.ckpt'

    # load dataset, get source image and destination image
    tdata, vdata = get_data()
    s, _ = tdata[3]
    d, _ = tdata[0]

    # load model
    lenet = LeNet()
    model = LeNetPL(lenet)
    model.load_from_checkpoint(checkpoint_path, lenet=lenet)
    find_path(s, d, model)


if __name__ == '__main__':
    main()
