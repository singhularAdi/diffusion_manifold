import torch
from torch.linalg import norm


def project_on_one(x, y):
    '''Project x onto y'''
    mag = torch.dot(x, y)
    vec = mag * (y / torch.linalg.norm(y))
    return vec


def project_subspace(delta, j):
    proj = torch.zeros_like(delta.view(-1))

    for i in range(j.shape[0]):
        proj += project_on_one(delta.view(-1), j[i].view(-1))

    return proj


def get_eigen(x, n_eigs=None):
    if n_eigs:
        eigvals, eigvecs = torch.lobpcg(x, k=n_eigs)
        eigvals, eigvecs = eigvals.real, eigvecs.real
    else:
        # Columns are eigvecs, project on transpose
        eigvals, eigvecs = torch.linalg.eig(x)
        eigvals, eigvecs = eigvals.real, eigvecs.real

        # Uncomment this section to plot eigaenvalues vs idx
        # sorted_vals, _ = eigvals.sort(descending=True)
        # plt.semilogy([idx for idx in range(len(eigvals))], sorted_vals)
        # plt.xlabel('idx (sorted)')
        # plt.ylabel('eigenvalue')
        # plt.savefig('./assets/g_eigval_plot.png')

    return eigvals, eigvecs


def get_orthnorm_vec(span, device='cpu'):
    x = torch.randn(span.shape[1], device=device)
    x_proj = project_subspace(x, span)
    x = x - x_proj
    return x / norm(x)
