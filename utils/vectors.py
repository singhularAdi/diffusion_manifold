import torch


def project_on_one(x, y):
    '''Project x onto y'''
    mag = torch.dot(x, y)
    vec = mag * (y / torch.linalg.norm(y))
    return vec


def gram_schmidt_orthogonalize():
    ...


def project_subspace(delta, j):
    proj = torch.zeros_like(delta.view(-1))

    for i in range(j.shape[0]):
        proj += project_on_one(delta.view(-1), j[i].view(-1))

    return proj


def gram_schmidt(vv):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    nk = vv.size(0)
    uu = torch.zeros_like(vv, device=vv.device)
    uu[:, 0] = vv[:, 0].clone()
    for k in range(1, nk):
        vk = vv[k].clone()
        uk = 0
        for j in range(0, k):
            uj = uu[:, j].clone()
            uk = uk + projection(uj, vk)
        uu[:, k] = vk - uk
    for k in range(nk):
        uk = uu[:, k].clone()
        uu[:, k] = uk / uk.norm()
    return uu


if __name__ == '__main__':
    import ipdb; ipdb.set_trace()
    a = torch.randn(5, 5, requires_grad=True)

    b = gram_schmidt(a)
    c = b.sum()
    c.backward()
    print(b.matmul(b.t()))
    print(a.grad)
