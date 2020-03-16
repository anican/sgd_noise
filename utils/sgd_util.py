import gc
import numpy as np
import torch


def write_tail_data(tail_data, tail_path: str):
    """
    Writes the given data to the given path in a format such that it can be reconstructed for future analysis.

    :param tail_data: 3-tuple of list of tensors (iterations, norms, alphas)
    :param tail_path: file path where the tail-index relevant data is stored
    :return:
    """
    iterations, grad_norms, alphas = zip(*tail_data)
    tail_data_history = {"Iterations": iterations, "SGD Norms":grad_norms, "Alpha Estimates":alphas}
    torch.save(tail_data_history, tail_path)


def get_tail_index(sgd_noise):
    """
    Returns an estimate of the tail-index term of the alpha-stable distribution for the stochastic gradient noise.
    In the paper, the tail-index is denoted by $\alpha$. Simsekli et. al. use the estimator posed by Mohammadi et al. in
    2015.

    :param sgd_noise:
    :return: tail-index term ($\alpha$) for an alpha-stable distribution
    """
    X = sgd_noise.reshape(-1)
    X = X[X.nonzero()]
    K = len(X)
    if len(X.shape)>1:
        X = X.squeeze()
    K1 = int(np.floor(np.sqrt(K)))
    K2 = K1
    X = X[:K1*K2].reshape((K2, K1))
    Y = X.sum(1)
    # X = X.cpu().clone(); Y = Y.cpu().clone()
    a = torch.log(torch.abs(Y)).mean()
    b = (torch.log(torch.abs(X[:K2/4,:])).mean()+torch.log(torch.abs(X[K2/4:K2/2,:])).mean()+torch.log(torch.abs(X[K2/2:3*K2/4,:])).mean()+torch.log(torch.abs(X[3*K2/4:,:])).mean())/4
    alpha_hat = np.log(K1)/(a-b).item()
    return alpha_hat


def get_sgd_noise(model, arch_type, curr_device, opt, full_loader):
    """

    :param model:
    :param arch_type:
    :param curr_device:
    :param opt:
    :param full_loader:
    :return:
    """
    gc.collect()
    # We do NOT want to be training on the full gradients, just calculating them!!!!
    model.eval()
    grads, sizes = [], []
    for batch_idx, (inputs, labels) in enumerate(full_loader):
        inputs, labels = inputs.to(curr_device), labels.to(curr_device)
        opt.zero_grad()
        if arch_type == 'mlp':
            inputs = inputs.view(inputs.size(0), -1)
        outputs = model(inputs)
        loss = model.loss(outputs, labels)
        loss.backward()
        grad = [param.grad.cpu().clone() for param in model.parameters()]
        # grad = [p.grad.clone() for p in model.parameters()]
        size = inputs.shape[0]
        grads.append(grad)
        sizes.append(size)

    flat_grads = []
    for grad in grads:
        flat_grads.append(torch.cat([g.reshape(-1) for g in grad]))
    full_grads = torch.zeros(flat_grads[-1].shape)
    # Exact_Grad = torch.zeros(Flat_Grads[-1].shape).cuda()
    for g, s in zip(flat_grads, sizes):
        full_grads += g * s
    full_grads /= np.sum(sizes)
    gc.collect()
    flat_grads = torch.stack(flat_grads)
    sgd_noise = (flat_grads-full_grads).cpu()
    # Grad_noise = Flat_Grads-Exact_Grad
    return full_grads, sgd_noise
