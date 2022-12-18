import torch
from functorch import jacrev
from torch.nn.functional import log_softmax
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_cluster import knn

from utils import get_eigen, get_orthnorm_vec, prep_img_for_classifier
from models import Diffusion, UNet_conditional


def predict(model, x):
    pred = model(x)
    return log_softmax(pred)

def _compute_g(x_orig, model, num_classes):

    x_orig_c = prep_img_for_classifier(x_orig)

    j = jacrev(predict, argnums=1)(model.lenet, x_orig_c.unsqueeze(0))
    j = j.squeeze()
    g = j.view(num_classes, -1).T @ j.view(num_classes, -1)
    return g


def _compute_g_perp(x_orig, model, num_classes, n_eigs, device):

    g = _compute_g(x_orig, model, num_classes)
    eigvals, eigvecs = get_eigen(g, n_eigs)

    g_perp = get_orthnorm_vec(eigvecs.T, device=device)
    g_perp = g_perp.to(device)

    return g_perp


def _compute_g_perp_avg(x_subset, closest_idxs, model, num_classes, n_eigs, device):
    print(x_subset.shape)
    g_all = torch.zeros(len(closest_idxs), 28 * 28, 28 * 28)
    for idx, data_idx in enumerate(closest_idxs):
        g_all[idx] = _compute_g(x_subset[idx].view(1, 64, 64), model, num_classes)

    g_avg = g_all.mean(axis=0)
    g_avg = g_avg.to(device)
    eigvals, eigvecs = get_eigen(g_avg, n_eigs)

    g_perp = get_orthnorm_vec(eigvecs.T, device=device)

    g_perp = g_perp.to(device)

    return g_perp




def compute_g_perp(
    x_orig, x_subset, model, num_classes, n_eigs, device, use_local_g
):

    dim = x_orig.shape[1]
    if not use_local_g:
        g_perp = _compute_g_perp(x_orig, model, num_classes, n_eigs, device)

    elif use_local_g:
        x_subset = x_subset.cpu()
        x_orig = x_orig.cpu()
        closest_idxs = knn(x_subset.view(-1, dim * dim),
                           x_orig.view(-1, dim * dim), k=10)
        closest_idxs = closest_idxs[1, :]

        x_orig = x_orig.to(device)
        x_subset = x_subset.to(device)
        g_perp = _compute_g_perp_avg(x_subset, closest_idxs, model, num_classes,
                                n_eigs, device)


    return g_perp


def find_path_between_manifolds(
    model, x_orig, x_subset, label=5, T=1000, num_classes=10, n_eigs=100,
    use_local_g=False, device='cuda'
):

    dif_backbone = UNet_conditional(c_in=1, c_out=1, num_classes=10).to(device)
    dif_backbone.load_state_dict(torch.load('lightning_logs/ema_ckpt_cond.pt'))
    diffusion = Diffusion(img_size=64, device=device)
    dist_hist = []

    g_perp = compute_g_perp(x_orig, x_subset, model, num_classes, n_eigs, device,
                                use_local_g)

    x_orig = x_orig.to(device)
    x_orig = x_orig.view(1, 1, 64, 64)
    label = torch.tensor([label]).to(device)
    t = torch.tensor([999]).to(device)

    noised_img = diffusion.noise_images(x_orig, t)

    sam, dist_hist= diffusion.sample_across(dif_backbone, 1, label, g_perp, x=noised_img[0], cfg_scale=0)
    import ipdb; ipdb.set_trace()


    plt.plot([[idx] for idx in range(len(dist_hist))], dist_hist)




