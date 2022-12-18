import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import normalize_to_neg_one_to_one

from scripts import find_path_on_manifold, find_path_between_manifolds
from utils import get_data, get_data_diff, prep_img_for_classifier
from models import LeNet, LeNetPL, UNet_conditional, Diffusion


def imshow_torch(img, shape=(28, 28)):
    plt.imshow(img.view(shape).detach().cpu().numpy())
    plt.show()



def gt_data():
    tdata, vdata = get_data()
    #s, _ = tdata[3]
    #d, _ = tdata[0]

    s, _ = tdata[0]
    d, _ = tdata[420]

    tdatadiff, vdatadiff = get_data_diff()

    s_diff, lab_diff = tdatadiff[0]
    diff_idxs = tdatadiff.targets == lab_diff
    diff_idxs = torch.where(diff_idxs == True)[0]

    diff_subset = torch.empty(len(diff_idxs), 64 * 64)

    for idx, idx_data in tqdm(enumerate(diff_idxs)):
        diff_subset[idx] = tdatadiff[idx_data][0].view(1, -1)

    return s, d, s_diff, diff_subset


def get_diffusion_unnoised(img, label=5, device='cuda', conditional=False):
    # Get model

    model = UNet_conditional(c_in=1, c_out=1, num_classes=10).to(device)
    model.load_state_dict(torch.load('lightning_logs/ema_ckpt_cond.pt'))
    diffusion = Diffusion(img_size=64, device=device)

    img = img.to(device)
    img = img.view(1, 1, 64, 64)
    label = torch.tensor([label]).to(device)
    t = torch.tensor([999]).to(device)

    noised_img = diffusion.noise_images(img, t)

    sam = diffusion.sample(model, 1, label, x=noised_img[0], cfg_scale=0)
    img = prep_img_for_classifier(img)
    diff_img = prep_img_for_classifier(sam[0]/255)

    return img.view(1, 28, 28), diff_img.view(1, 28, 28)



def main():

    device = 'cuda'
    checkpoint_path = '/home/aditya/workspace/diffusion_manifold/lightning_logs/version_0/checkpoints/lmnist-epoch=20-val_loss=0.0303-val_acc=0.9914.ckpt'

    # get gt data
    s, d, s_diff, diff_subset = gt_data()
    s, d, s_diff = s.to(device), d.to(device), s_diff.to(device)

    # load model
    lenet = LeNet()
    model = LeNetPL(lenet)
    model.load_from_checkpoint(checkpoint_path, lenet=lenet)
    model.to(device)
    # Natural Data
    #find_path_on_manifold(s, d, model, use_g=True, T=1000, log_steps=100, save_path=True)  # norm_dis = 0.133, reim dist t = 1000
    # find_path_on_manifold(s, d, model, use_g=True, T=1000, log_steps=100, n_eigs=50, save_path=True)  # n = 8.45, r_dis = 1000
    # find_path_on_manifold(s, d, model, use_g=True, T=10000, log_steps=1000, n_eigs=10, save_path=True)  # n = 4.14, r_dis = 10000
    # find_path_on_manifold(s, d, model, alpha=0.1,  use_g=False, T=10000, log_steps=1000,  save_path=True)  # norm dist= 10.5, reim dist trvelled = 10000

    #find_path_on_manifold(s, d, model, alpha=0.1,  use_g=True, T=1000, log_steps=100, n_eigs=50, save_path=True)  # norm dist= 10.5, reim dist trvelled = 10000

    s_diff_r, s_denoised = get_diffusion_unnoised(s_diff, device=device)
    # Diffusion Data

    #find_path_on_manifold(s_diff_r, s_denoised, model, use_g=False, T=10000, log_steps=1000, save_path=True)  # n = 15.9, r_dis = 1000

    #find_path_on_manifold(s_diff_r, s_denoised, model, use_g=True, T=10000, log_steps=1000, n_eigs=10, save_path=True)  # n = 15.9, r_dis = 1000
    #find_path_on_manifold(s_diff_r, s_denoised, model, use_g=True, T=1000, log_steps=100, n_eigs=100, save_path=True)  # n = 15.9, r_dis = 1000
    #find_path_on_manifold(s_diff_r, s_denoised, model, use_g=True, T=1000, log_steps=100, n_eigs=None, save_path=True)  # n = 15.9, r_dis = 1000
    # Across traversal
    find_path_between_manifolds(model, s_diff, diff_subset, use_local=True, use_g=False)


if __name__ == '__main__':
    main()
