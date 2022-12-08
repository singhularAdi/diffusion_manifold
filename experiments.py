from scripts import find_path
from utils import get_data
from models import LeNet, LeNetPL


def gt_data():
    tdata, vdata = get_data()
    s, _ = tdata[3]
    #d, _ = tdata[0]

    #s, _ = tdata[0]
    d, _ = tdata[420]

    return s, d


def main():

    checkpoint_path = '/home/aditya/workspace/diffusion_manifold/lightning_logs/version_0/checkpoints/lmnist-epoch=20-val_loss=0.0303-val_acc=0.9914.ckpt'

    # get gt data
    s, d = gt_data()

    # load model
    lenet = LeNet()
    model = LeNetPL(lenet)
    model.load_from_checkpoint(checkpoint_path, lenet=lenet)
    #find_path(s, d, model, use_g=True, T=1000, log_steps=100, n_eigs=50, save_path=True)
    find_path(s, d, model, alpha=0.1,  use_g=False, T=10000, log_steps=1000,  save_path=True)


if __name__ == '__main__':
    main()
