import torch
from functorch import jacrev
from torch.nn.functional import log_softmax
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import get_eigen, get_orthnorm_vec, prep_img_for_classifier
from models import Diffusion, UNet_conditional


def predict(model, x):
    pred = model(x)
    return log_softmax(pred)


def find_path_between_manifolds(model, x_orig, label=5,
                     T=1000, num_classes=10, n_eigs=100, device='cuda'):

    dif_backbone = UNet_conditional(c_in=1, c_out=1, num_classes=10).to(device)
    dif_backbone.load_state_dict(torch.load('lightning_logs/ema_ckpt_cond.pt'))
    diffusion = Diffusion(img_size=64, device=device)
    dist_hist = []

    x_orig_c = prep_img_for_classifier(x_orig)
    j = jacrev(predict, argnums=1)(model.lenet, x_orig_c.unsqueeze(0))
    j = j.squeeze()
    g = j.view(num_classes, -1).T @ j.view(num_classes, -1)
    eigvals, eigvecs = get_eigen(g, n_eigs)

    g_perp = get_orthnorm_vec(eigvecs.T, device=device)
    g_perp = g_perp.to(device)


    x_orig = x_orig.to(device)
    x_orig = x_orig.view(1, 1, 64, 64)
    label = torch.tensor([label]).to(device)
    t = torch.tensor([999]).to(device)

    noised_img = diffusion.noise_images(x_orig, t)

    sam, dist_hist= diffusion.sample_across(dif_backbone, 1, label, g_perp, x=noised_img[0], cfg_scale=0)
    import ipdb; ipdb.set_trace()


#    plt.plot([[idx] for idx in range(len(dist_hist))], dist_host)




