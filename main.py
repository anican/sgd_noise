import argparse
from models import GoogLeNet
import numpy as np
from project import Project
import time
import torch
from torch import multiprocessing
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import traceback
from utils import pt_util


def train(model, device, data_loader, optimizer, epoch, log_interval):
    model.train()
    losses = []
    for batch_idx, (data, label) in enumerate(data_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = model.loss(output, label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time.ctime(time.time()),
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))
    return np.mean(losses)


def test(model, device, data_loader, return_images=False, log_interval=None):
    model.eval()
    test_loss, correct = 0, 0
    correct_images, correct_values, error_images, predicted_values, gt_values = [], [], [], [], []

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(data_loader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            test_loss_on = model.loss(output, label, reduction='sum').item()
            test_loss += test_loss_on
            prediction = output.max(1)[1]
            correct_mask = prediction.eq(label.view_as(prediction))
            num_correct = correct_mask.sum().item()
            correct += num_correct
            if return_images:
                if num_correct > 0:
                    correct_images.append(data[correct_mask, ...].data.cpu().numpy())
                    correct_value_data = label[correct_mask].data.cpu().numpy()[:, 0]
                    correct_values.append(correct_value_data)
                if num_correct < len(label):
                    error_data = data[~correct_mask, ...].data.cpu().numpy()
                    error_images.append(error_data)
                    predicted_value_data = prediction[~correct_mask].data.cpu().numpy()
                    predicted_values.append(predicted_value_data)
                    gt_value_data = label[~correct_mask].data.cpu().numpy()[:, 0]
                    gt_values.append(gt_value_data)
            if log_interval is not None and batch_idx % log_interval == 0:
                print('{} Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    time.ctime(time.time()),
                    batch_idx * len(data), len(data_loader.dataset),
                    100. * batch_idx / len(data_loader), test_loss_on))
    if return_images:
        correct_images = np.concatenate(correct_images, axis=0)
        error_images = np.concatenate(error_images, axis=0)
        predicted_values = np.concatenate(predicted_values, axis=0)
        correct_values = np.concatenate(correct_values, axis=0)
        gt_values = np.concatenate(gt_values, axis=0)

    test_loss /= len(data_loader.dataset)
    test_accuracy = 100. * correct / len(data_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset), test_accuracy))
    if return_images:
        return test_loss, test_accuracy, correct_images, correct_values, error_images, predicted_values, gt_values
    else:
        return test_loss, test_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='<Add Project Description>')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--print_interval', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    project = Project()

    # Data Transforms and Datasets
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    data_train = datasets.CIFAR10(root=str(project.DATA_PATH), train=True, download=True, transform=transform_train)
    data_test = datasets.CIFAR10(root=str(project.DATA_PATH), train=False, download=True, transform=transform_test)

    # Train on GPU (if CUDA is available)
    # torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    torch_device = torch.device("cuda" if use_cuda else "cpu")
    print('using device', torch_device)

    # Neural Network Model
    network = GoogLeNet().to(torch_device)

    # Loss Functions TODO:
    criterion = nn.MSELoss()

    # Create Optimizer
    opt = optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    num_workers = multiprocessing.cpu_count()
    print('Number of CPUs:', num_workers)
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # replace with get_dataloaders() in the template overall
    train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(data_test, batch_size=args.test_batch_size, **kwargs)

    start_epoch = network.load_last_model(str(project.WEIGHTS_PATH))
    train_losses, test_losses, test_accuracies = pt_util.read_log(str(project.LOG_PATH), ([], [], []))
    test_loss, test_accuracy = test(network, torch_device, test_loader)
    test_losses.append((start_epoch, test_loss))
    test_accuracies.append((start_epoch, test_accuracy))

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss = train(network, torch_device, train_loader, opt, epoch, args.print_interval)
            test_loss, test_accuracy = test(network, torch_device, test_loader)
            train_losses.append((epoch, train_loss))
            test_losses.append((epoch, test_loss))
            test_accuracies.append((epoch, test_accuracy))
            pt_util.write_log(str(project.LOG_PATH), (train_losses, test_losses, test_accuracies))
            network.save_best_model(test_accuracy, str(project.WEIGHTS_PATH) + '/%03d.pt' % epoch)
    except KeyboardInterrupt as ke:
        print('Manually interrupted execution...')
    except:
        traceback.print_exc()
    finally:
        # TODO: Shouldn't this be saved to most recent epoch
        print('Saving model in its current state')
        network.save_model(str(project.WEIGHTS_PATH) + '/%03d.pt' % epoch, 0)
        ep, val = zip(*train_losses)
        pt_util.plot(ep, val, 'Train loss', 'Epoch', 'Error')
        ep, val = zip(*test_losses)
        pt_util.plot(ep, val, 'Test loss', 'Epoch', 'Error')
        ep, val = zip(*test_accuracies)
        pt_util.plot(ep, val, 'Test accuracy', 'Epoch', 'Error')

