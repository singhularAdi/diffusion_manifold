from torch.linalg import norm
from torch.nn.functional import log_softmax
from functorch import jacrev

from utils import get_data, project_subspace, gram_schmidt_orthogonalize
from models import LeNet, LeNetPL


def predict(model, x):
    pred = model(x)
    return log_softmax(pred)


def find_path(s, d, model, alpha=0.1, T=5000):
    x_t = s

    for t in range(T):
        # calculate jacobian of log_softmax wrt x_t
        j = jacrev(predict, argnums=1)(model.lenet, x_t.unsqueeze(0))
        j = j.squeeze()

        # orthonomalize for projection
        j = gram_schmidt_orthogonalize(j)
        # project d-x_t onto j
        v = project_subspace(d - x_t, j)

        # update
        x_t = x_t + alpha * (v / norm(v))


def main():

    checkpoint_path = '/home/aditya/workspace/diffusion_manifold/lightning_logs/version_0/checkpoints/lmnist-epoch=20-val_loss=0.0303-val_acc=0.9914.ckpt'

    # load dataset, get source image and destination image
    tdata, vdata = get_data()
    s, sl = tdata[0]
    d, dl = tdata[1]

    # load model
    lenet = LeNet()
    model = LeNetPL(lenet)
    model.load_from_checkpoint(checkpoint_path, lenet=lenet)
    find_path(s, d, model)


if __name__ == '__main__':
    main()
