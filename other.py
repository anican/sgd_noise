import time
import torch


def test(model, device, data_loader, log_interval=None):
    """
    Less fancy version of 'test' method
    :param model:
    :param device:
    :param data_loader:
    :param log_interval:
    :return:
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(data_loader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            test_loss_on = model.loss(output, label, reduction='sum').item()
            test_loss += test_loss_on
            pred = output.max(1)[1]
            correct_mask = pred.eq(label.view_as(pred))
            num_correct = correct_mask.sum().item()
            correct += num_correct
            if log_interval is not None and batch_idx % log_interval == 0:
                print('{} Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    time.ctime(time.time()),
                    batch_idx * len(data), len(data_loader.dataset),
                    100. * batch_idx / len(data_loader), test_loss_on))

    test_loss /= len(data_loader.dataset)
    test_accuracy = 100. * correct / len(data_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset), test_accuracy))
    return test_loss, test_accuracy