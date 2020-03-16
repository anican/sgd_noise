import argparse
from models import MLP, AlexNet
from models import linear_hinge_loss
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


def train(model, arch_type, curr_device, data_loader, optimizer, curr_epoch, log_interval):
    model.train()
    losses = []
    for batch_idx, (inputs, label) in enumerate(data_loader):
        inputs, label = inputs.to(curr_device), label.to(curr_device)
        optimizer.zero_grad()
        if arch_type == 'mlp':
            inputs = inputs.view(inputs.size(0), -1)
        outputs = model(inputs)
        loss = model.loss(outputs, label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time.ctime(time.time()),
                curr_epoch, batch_idx * len(inputs), len(data_loader.dataset), 100. * batch_idx / len(data_loader),
                loss.item()))
    return np.mean(losses)


def test(model, arch_type, curr_device, data_loader, return_images=False, log_interval=None):
    model.eval()
    test_loss, correct = 0, 0
    correct_images, correct_values, error_images, predicted_values, gt_values = [], [], [], [], []

    with torch.no_grad():
        for batch_idx, (inputs, label) in enumerate(data_loader):
            inputs, label = inputs.to(curr_device), label.to(curr_device)
            if arch_type == 'mlp':
                inputs = inputs.view(inputs.size(0), -1) 
            outputs = model(inputs)
            # TODO: if the output here is messed up change loss to be from torch.nn.functional to accomodate change in
            #       reduction or just multiply by batch_size?
            # test_loss_on = model.loss(outputs, label, reduction='sum').item()
            test_loss_on = model.loss(outputs, label).item()
            test_loss += test_loss_on
            prediction = outputs.max(1)[1]
            correct_mask = prediction.eq(label.view_as(prediction))
            num_correct = correct_mask.sum().item()
            correct += num_correct
            if return_images:
                if num_correct > 0:
                    correct_images.append(inputs[correct_mask, ...].data.cpu().numpy())
                    correct_value_data = label[correct_mask].data.cpu().numpy()[:, 0]
                    correct_values.append(correct_value_data)
                if num_correct < len(label):
                    error_data = inputs[~correct_mask, ...].data.cpu().numpy()
                    error_images.append(error_data)
                    predicted_value_data = prediction[~correct_mask].data.cpu().numpy()
                    predicted_values.append(predicted_value_data)
                    gt_value_data = label[~correct_mask].data.cpu().numpy()[:, 0]
                    gt_values.append(gt_value_data)
            if log_interval is not None and batch_idx % log_interval == 0:
                print('{} Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    time.ctime(time.time()),
                    batch_idx * len(inputs), len(data_loader.dataset),
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
    parser = argparse.ArgumentParser(description='SGD Noise Experiments')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--levy_alpha', type=float, default=-1.0, help='tail index of added levy motion')
    parser.add_argument('--levy_sigma', type=float, default=-1.0, help='scale parameter of added levy noise')
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--neurons', type=int, default=1024)  # number of neurons in hidden layer
    parser.add_argument('--print_interval', type=int, default=100)
    parser.add_argument('--criterion', type=str, default="cross_entropy")  # 'cross_entropy' or 'linear_hinge'
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--architecture', type=str, default='mlp')
    args = parser.parse_args()
    project = Project()

    # Data Transforms and Dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    data_train = datasets.CIFAR10(root=str(project.DATA_PATH), train=True, download=True, transform=transform_train)
    data_test = datasets.CIFAR10(root=str(project.DATA_PATH), train=False, download=True, transform=transform_test)
    train_size = data_train.data.shape[0]
    test_size = data_test.data.shape[0]

    # Train on GPU (if CUDA is available)
    # torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device:', device)

    # Loss Functions
    if args.criterion == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif args.criterion == 'linear_hinge':
        criterion = linear_hinge_loss
    else:
        raise ValueError("Invalid criterion selection!")
    print("Criterion:", criterion)

    # Neural Network Model
    if args.architecture == 'mlp':
        network = MLP(criterion).to(device)
    elif args.architecture == 'alex_net':
        network = AlexNet(criterion).to(device)
    else:
        raise ValueError("Invalid model selection!")
    print("Model:", network)

    # Create Optimizer
    opt = optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    print("Optimizer:", opt)

    num_workers = multiprocessing.cpu_count()
    print('Number of CPUs:', num_workers)
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # replace with get_dataloaders() in the template overall
    train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(data_test, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # TODO: Why is the batch size for full loader 4096 in cifar_main.py
    train_loader_full = DataLoader(data_train, batch_size=train_size, shuffle=False, **kwargs)
    test_loader_full = DataLoader(data_test, batch_size=test_size, shuffle=False, **kwargs)

    start_epoch = network.load_last_model(str(project.WEIGHTS_PATH))
    train_losses, test_losses, test_accuracies = pt_util.read_log(str(project.LOG_PATH), ([], [], []))
    test_loss, test_accuracy = test(network, device, test_loader)
    test_losses.append((start_epoch, test_loss))
    test_accuracies.append((start_epoch, test_accuracy))

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss = train(model=network, arch_type=args.architecture, curr_device=device, data_loader=train_loader,
                               optimizer=opt, curr_epoch=epoch, log_interval=args.print_interval)
            test_loss, test_accuracy = test(model=network, arch_type=args.architecture, curr_device=device,
                                            data_loader=test_loader)
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

