# Define custom loss functions
import torch
import torch.nn.functional as F


def linear_hinge_loss(output, target):
    binary_target = output.new_empty(*output.size()).fill_(-1)
    for i in range(len(target)):
        binary_target[i, target[i]] = 1
    delta = 1 - binary_target * output
    delta[delta <= 0] = 0
    return delta.mean()


def square_loss(output, target):
    loss_val = torch.mean((output - target)**2)
    return loss_val


if __name__ == '__main__':
    output = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, dtype=torch.bool).random_(5)
    loss = square_loss(output, target)
    loss.backward()


