import argparse
from models import MLP, AlexNet
from models import linear_hinge_loss
import numpy as np
from paths import ProjectPaths
import time
import torch
from torch import multiprocessing
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import traceback
from utils import pt_util, sgd_util


def train(model, arch_type, curr_device, data_loader, full_loader, optimizer, curr_epoch, log_interval, tail_data):
    model.train()
    losses = []
    for batch_idx, (inputs, label) in enumerate(data_loader):
        curr_iteration = len(data_loader) * curr_epoch + batch_idx + 1
        inputs, label = inputs.to(curr_device), label.to(curr_device)
        optimizer.zero_grad()
        if arch_type == 'mlp':
            inputs = inputs.view(inputs.size(0), -1)
        outputs = model(inputs)
        loss = model.loss(outputs, label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        full_grad_train, sgd_noise_train = sgd_util.get_sgd_noise(model=model, arch_type=arch_type,
                                                                  curr_device=curr_device, opt=optimizer,
                                                                  full_loader=full_loader)
        alpha_train = sgd_util.get_tail_index(sgd_noise=sgd_noise_train)
        tail_data.append((curr_iteration, torch.norm(sgd_noise_train, dim=1), alpha_train))
        if batch_idx % log_interval == 0:
            print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time.ctime(time.time()),
                curr_epoch, batch_idx * len(inputs), len(data_loader.dataset), 100. * batch_idx / len(data_loader),
                loss.item()))
    return np.mean(losses)


def test(model, arch_type, curr_device, data_loader, return_images=False, log_interval=None):
    model.eval()
    loss, correct = 0, 0
    correct_images, correct_values, error_images, predicted_values, gt_values = [], [], [], [], []

    with torch.no_grad():
        for batch_idx, (inputs, label) in enumerate(data_loader):
            inputs, label = inputs.to(curr_device), label.to(curr_device)
            if arch_type == 'mlp':
                inputs = inputs.view(inputs.size(0), -1)
            outputs = model(inputs)
            # TODO: if output here is messed up change loss to be from torch.nn.functional to acc change in reduction
            # test_loss_on = model.loss(outputs, label, reduction='sum').item()
            test_loss_on = model.loss(outputs, label).item()
            loss += test_loss_on
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

    loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss, correct, len(data_loader.dataset), accuracy))
    if return_images:
        return loss, accuracy, correct_images, correct_values, error_images, predicted_values, gt_values
    else:
        return loss, accuracy


def get_arguments():
    parser = argparse.ArgumentParser(description='SGD Noise Experiments')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=64)
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
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    project_paths = ProjectPaths()

    # Data Transforms and Dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    data_train = datasets.CIFAR10(root=str(project_paths.DATA_PATH), train=True, download=True, transform=transform_train)
    data_test = datasets.CIFAR10(root=str(project_paths.DATA_PATH), train=False, download=True, transform=transform_test)
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

    train_loader_full = DataLoader(data_train, batch_size=train_size, shuffle=False, **kwargs)
    test_loader_full = DataLoader(data_test, batch_size=test_size, shuffle=False, **kwargs)

    start_epoch = network.load_last_model(str(project_paths.WEIGHTS_PATH))
    train_losses, test_losses, test_accuracies = pt_util.read_log(str(project_paths.LOG_PATH), ([], [], []))
    test_loss, test_accuracy = test(model=network, arch_type=args.architecture, curr_device=device,
                                    data_loader=test_loader)
    test_losses.append((start_epoch, test_loss))
    test_accuracies.append((start_epoch, test_accuracy))

    # Stores gradient norms and alphas for each batch iteration during training :)!!
    train_tail_data = []

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss = train(model=network, arch_type=args.architecture, curr_device=device, data_loader=train_loader,
                               full_loader=train_loader_full, optimizer=opt, curr_epoch=epoch,
                               log_interval=args.print_interval, tail_data=train_tail_data)
            test_loss, test_accuracy = test(model=network, arch_type=args.architecture, curr_device=device,
                                            data_loader=test_loader)
            train_losses.append((epoch, train_loss))
            test_losses.append((epoch, test_loss))
            test_accuracies.append((epoch, test_accuracy))
            pt_util.write_log(str(project_paths.LOG_PATH), (train_losses, test_losses, test_accuracies))
            sgd_util.write_tail_data(train_tail_data, str(project_paths.TAILS_PATH) + '/tails%03d.pt' % epoch)
            network.save_best_model(test_accuracy, str(project_paths.WEIGHTS_PATH) + '/weights%03d.pt' % epoch)
    except KeyboardInterrupt as ke:
        print('Manually interrupted execution...')
    except:
        traceback.print_exc()
    finally:
        print('Saving model in its current state')
        network.save_model(str(project_paths.WEIGHTS_PATH) + '/%03d.pt' % epoch, 0)
        ep, val = zip(*train_losses)
        pt_util.plot(ep, val, 'Train loss', 'Epoch', 'Error')
        ep, val = zip(*test_losses)
        pt_util.plot(ep, val, 'Test loss', 'Epoch', 'Error')
        ep, val = zip(*test_accuracies)
        pt_util.plot(ep, val, 'Test accuracy', 'Epoch', 'Error')
