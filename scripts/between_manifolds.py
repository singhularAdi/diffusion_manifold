import torch
from functorch import jacrev
from torch.nn.functional import log_softmax
from tqdm import trange

from utils import project_subspace, get_eigen, get_orthnorm_vec


def predict(model, x):
    pred = model(x)
    return log_softmax(pred)


@torch.no_grad()
def diffusion_traversal(model, x_orig, x_noisy, T=1000, num_classes=10, n_eigs=100):

    dist_hist = []

    j = jacrev(predict, argnums=1)(model.lenet, x_orig.unsqueeze(0))
    j = j.squeeze()
    g = j.view(num_classes, -1).T @ j.view(num_classes, -1)
    eigvals, eigvecs = get_eigen(g, n_eigs)

    g_perp = get_orthnorm_vec(eigvecs.T)

    # T should be the same as no. of diffusion steps
    with trange(T, desc='Undiffusion') as pbar:

        x_new = ...  # move one step using diffusion model

        delta_x = x_new - x_noisy

        perp_dis = torch.dot(g_perp, delta_x)

        dist_hist.append(perp_dis.detach().cpu().numpy())




