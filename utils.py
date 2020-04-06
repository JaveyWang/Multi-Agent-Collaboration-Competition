import torch


def R1_penalty(out, samples, reg_lambda=10):
    """
    private helper for calculating the gradient penalty
    :param out: real samples
    :param reg_lambda: regularisation lambda
    :return: gradient_penalty => scalar tensor
    """
    from torch.autograd import grad

    batch_size = samples.shape[0]

    # obtain gradient of op wrt. merged
    gradient = grad(outputs=out, inputs=samples, create_graph=True,
                    grad_outputs=torch.ones_like(out),
                    retain_graph=True, only_inputs=True)[0]

    # calculate the penalty using these gradients
    penalty = reg_lambda * (gradient.view(batch_size, -1).pow(2).sum(1)).mean()

    # return the calculated penalty:
    return penalty

def get_masks(lengths):
    masks = torch.zeros([len(lengths), max(lengths)])
    for i, l in enumerate(lengths):
        masks[i, :l] = 1
    return masks