import torch
from torch import nn
from utils import pt_util


def num_flat_features(x: torch.Tensor) -> int:
    r"""

    :param x:
    :return:

    Examples::
        >>> x = torch.randn(5, 3, 7, 7)
        >>> num_flat_features(x)
        147
    """
    num_features = 1
    for dim in x.size()[1:]:
        num_features *= dim
    return num_features


class BaseModule(nn.Module):
    r"""
    Base module for all neural network models. Adds additional functionality for fancy 'restore' during neural network
    experiments.
    """
    def __init__(self):
        super(BaseModule, self).__init__()
        self.best_accuracy = 0

    def forward(self, *input):
        raise NotImplementedError

    def loss(self, prediction, label, reduction='mean'):
        r"""
        Specifies the loss criterion for this neural network model.

        :param prediction:
        :param label: target labels associated with the prediction spit out by the model.
                      Must be of type torch.LongTensor.
        :param reduction:
        :return:

        Examples::
            >>> # sample implementation of loss() using nn.CrossEntropyLoss() from torch.nn
            >>> loss_fn = nn.CrossEntropyLoss(reduction=reduction)
            >>> loss = loss_fn(prediction, label.squeeze().long())
            >>> return loss
        """
        raise NotImplementedError

    def save_model(self, file_path, num_to_keep=1):
        pt_util.save(self, file_path, num_to_keep)

    def save_best_model(self, accuracy, file_path, num_to_keep=1):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.save_best_model(self, file_path, num_to_keep)

    def load_model(self, file_path):
        pt_util.restore(self, file_path)

    def load_last_model(self, log_path):
        return pt_util.restore_latest(self, log_path)


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)

