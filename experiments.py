import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import normalize_to_neg_one_to_one

from scripts import find_path, find_path_across
from utils import get_data, get_data_diff, prep_img_for_classifier
from models import LeNet, LeNetPL


def imshow_torch(img, shape=(28, 28)):
    plt.imshow(img.view(shape).detach().cpu().numpy())
    plt.show()



def gt_data():
    tdata, vdata = get_data()
    s, _ = tdata[3]
    d, _ = tdata[0]

    #s, _ = tdata[0]
    #d, _ = tdata[420]

    tdatadiff, vdatadiff = get_data_diff()

    s_diff, _ = tdatadiff[123]
    return s, d, s_diff


def get_diffusion_unnoised(img, device='cuda', conditional=False):
    # Get model
    import ipdb; ipdb.set_trace()

    if not conditional:

#        model = Unet(
#            dim = 64,
#            dim_mults = (1, 2),
#            #num_classes=10,
#            channels = 1,
#            #cond_drop_prob = 0.1
#        )
#
#        diffusion = GaussianDiffusion(
#            model,
#            image_size = 64,
#            timesteps = 1000,   # number of steps
#        ).cuda()
#        diffusion.load_state_dict(torch.load('./lightning_logs/ema_unconditional.pth'))
        diffusion = torch.load('lightning_logs/diffusion-model.pt')
        diffusion.to(device)
    else:
        ...

    # Prep data
    num_timesteps = 1000
    norm_img = normalize_to_neg_one_to_one(img)
    norm_img = norm_img.to(device)
    noised_img = diffusion.q_sample(norm_img, t=torch.tensor([num_timesteps - 1], device=device)).view(1, 1, 64, 64)


    for t in tqdm(reversed(range(0, num_timesteps)), desc='sampling loop ts', total=num_timesteps):
        noised_img, x_start = diffusion.p_sample(noised_img, t, None)

    import ipdb; ipdb.set_trace()
    img = prep_img_for_classifier(img)
    diff_img = prep_img_for_classifier(noised_img)

    return img, diff_img



def main():

    device = 'cuda'
    checkpoint_path = '/home/aditya/workspace/diffusion_manifold/lightning_logs/version_0/checkpoints/lmnist-epoch=20-val_loss=0.0303-val_acc=0.9914.ckpt'

    # get gt data
    s, d, s_diff = gt_data()
    s, d, s_diff = s.to(device), d.to(device), s_diff.to(device)

    # load model
    lenet = LeNet()
    model = LeNetPL(lenet)
    model.load_from_checkpoint(checkpoint_path, lenet=lenet)
    model.to(device)
    # Natural Data
    # find_path(s, d, model, use_g=True, T=1000, log_steps=100, save_path=True)  # norm_dis = 0.133, reim dist t = 1000
    # find_path(s, d, model, use_g=True, T=1000, log_steps=100, n_eigs=50, save_path=True)  # n = 8.45, r_dis = 1000
    # find_path(s, d, model, use_g=True, T=10000, log_steps=1000, n_eigs=10, save_path=True)  # n = 4.14, r_dis = 10000
    # find_path(s, d, model, alpha=0.1,  use_g=False, T=10000, log_steps=1000,  save_path=True)  # norm dist= 10.5, reim dist trvelled = 10000

    import ipdb; ipdb.set_trace()
    s_diff, s_denoised = get_diffusion_unnoised(s_diff, device)
    # Diffusion Data

    import ipdb; ipdb.set_trace()
    find_path(s_diff, s_denoised, model, use_g=True, T=1000, log_steps=100, n_eigs=50, save_path=True)  # n = 8.45, r_dis = 1000

    # Across traversal
    find_path_across(model, diffusion, norm_s_diff, noised_s)


if __name__ == '__main__':
    main()
