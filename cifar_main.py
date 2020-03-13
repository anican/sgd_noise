
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
# import torchvision.models as models
from cifar_models import *

import numpy as np
import pandas as pd
import os, gc
import argparse
import matplotlib.pyplot as plt
from operator import itemgetter

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--max_epochs', default=8, type=int, help='number of epochs to train model for')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset to train on')
parser.add_argument('--load_checkpoint', action='store_true', default=False, help='resume training from checkpoint if arg is given')
parser.add_argument('--save_checkpoint', action='store_true', default=False, help='saves a checkpoint every epoch if arg is given')
parser.add_argument('--modelDir', default="./chkpt_model/", help="model_path given by Philly")
parser.add_argument('--model_dir', default="/mnt/t-rasom/chkpt_model/", help="model_path")
parser.add_argument('--dataDir', default="/tmp", help="data_path given by Philly")
parser.add_argument('--model_prefix', default="model", help="model prefix for this run - model checkpoint filename = modelDir/model_prefix-N.pt : checkpoint after N epochs")
parser.add_argument('--batch_size', default=512, type=int, help='batch_size')
parser.add_argument('--runs',default=1,type=int,help='number of independent runs')
parser.add_argument('--echo_freq',default=5,type=int,help='Echo frequency')
parser.add_argument('--batch_verbose',type=int,default=10,help='Number of batch iterations before printing running loss')
parser.add_argument('--logDir', default="/tmp", help="log_path given by Philly")
parser.add_argument('--stdoutDir', default="", help="stdout dir given by Philly")
parser.add_argument('--change_model',default="LeNet", help="change LeNet to another model")
parser.add_argument('--batch_norm',action='store_true',default=False,help='adds batch norm layers')
parser.add_argument('--lr',type=float,default=1.0,help='set learning rate')
parser.add_argument('--factor',type=float,default=0.0,help='Divide lr by 2^factor')
parser.add_argument('--momentum',type=float,default=0.0,help='Set momentum')
parser.add_argument('--seed',type=int,default=0,help='Set seed')
parser.add_argument('--subset',type=int,default=0,help='Train on subset of training data. 0 means full data')
parser.add_argument('--random_labels',action='store_true',default=False,help='Assign random labels to training data')
parser.add_argument('--random_data',action='store_true',default=False,help='Random data')
parser.add_argument('--classes',type=int,default=0,help='Select the first few classes')
parser.add_argument('--training',default="MSE",help="select objective loss")
parser.add_argument('--store_prob', action='store_true', default=False, help='Store probability vectors of all train points')
parser.add_argument('--neurons',type=int,default=1024,help='Neurons in hidden layers if args.change_model==MLP')
parser.add_argument('--layers',type=int,default=1,help='Layers of MLP if args.change_model==MLP')
parser.add_argument('--store_features', action='store_true', default=False, help='Store features from the penultimate layer')
parser.add_argument('--freeze',type=int,default=-1,help='Freeze n-1 layers and only train the last layer (clssifier)')
parser.add_argument('--perturbe_gain',type=float,default=0.0,help='extra gain in xaview initialization')
parser.add_argument('--compute_alpha',action='store_true',default=False,help='Compute alphas of grad noise for alpha-levy')
parser.add_argument('--replacement',action='store_true',default=False,help='with or without replacement sampling')
parser.add_argument('--levy_alpha',type=float,default=-1.0,help='tail index of added levy motion')
parser.add_argument('--levy_sigma',type=float,default=-1.0,help='scale parameter of added levy noise')
parser.add_argument('--compute_stoch_norm',type=int,default=-1,help='Compute stochastic grad norms with this batch_size')
parser.add_argument('--evaluate_all',action='store_true',default=False,help='Evaluate all kinds of metrics')
parser.add_argument('--evaluate_continuous',type=int,default=-1,help='Evaluate after every iteration till this integer, then after every 10 iterations. -1 means do not evaluate.')
parser.add_argument('--stop_when_trained',action='store_true',default=False,help='Stop at t^th epoch when (t-5)^th epoch has reached 100% train accuracy')

args = parser.parse_args()
data_dir = args.dataDir
model_dir = args.model_dir
model_prefix=args.model_prefix
model_prefix += "_lr"+str(args.factor)
if args.seed!=0:
    # set RNG seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
else:
    np.random.seed(seed=None)
    torch.manual_seed(int(np.random.rand() * 100))

# print all program args
print("Program Arguments:")
print(args)

# hyperparams
NUM_TEST_PTS = 10000 # as far as I can tell, not used at all
NUM_TRAIN_PTS = 50000 # only used in shuffling the data sampler
DATASET=torchvision.datasets.CIFAR100 if args.dataset=='cifar100' else torchvision.datasets.CIFAR10
NUM_CLASSES = 100 if args.dataset=='cifar100' else 10
MAX_EPOCHS = args.max_epochs
MODEL_CLASS = "VGG16"#models.vgg16()
BATCH_SIZE = args.batch_size
FULL_BATCH_SIZE = 4096
CHKPT_PATH=os.path.join(model_dir, model_prefix)
lr = args.lr
runs = args.runs
lr = args.lr/2**args.factor
args.training = args.training.strip()

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)

def perturbe_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m_copy = m.weight.data.detach().clone()
        noise = nn.init.xavier_uniform_(m_copy,gain=args.perturbe_gain)
        m.weight.data += noise #nn.init.xavier_uniform_(m.weight.data.shape,gain=args.perturbe_gain)#torch.zeros(m.weight.data.shape).normal_(0,args.perturbe_std).cuda()



# create model
def create_model(model_class,batch_norm=True):
    if args.change_model=="LeNet":
        net = LeNet(batch_norm,num_classes=NUM_CLASSES)
    elif args.change_model=="ConvFc":
        net = ConvFc(batch_norm)
    elif "VGG" in args.change_model:
        net = VGG(args.change_model,batch_norm,num_classes=NUM_CLASSES)
    elif "ResNet50" in args.change_model:
        net = ResNet50(num_classes=NUM_CLASSES)
    elif "ResNet18" in args.change_model:
        net = ResNet18(num_classes=NUM_CLASSES)
    elif "DenseNet" in args.change_model:
        net = densenet_cifar(num_classes=NUM_CLASSES)
    elif "MLP" in args.change_model:
        net = MLP(3*32*32,args.neurons,NUM_CLASSES,args.layers,batch_norm)
    net = net.to('cuda')
    net = torch.nn.DataParallel(net)
    net.apply(weights_init)
    cudnn.benchmark = True
    if args.training == "MSE":
        criterion = nn.MSELoss(reduction='none') # CrossEntropyLoss
    elif args.training == "CE":
        criterion = nn.CrossEntropyLoss()
    elif args.training == "SE":
        criterion = nn.MSELoss()
    criterion = criterion.cuda()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=args.momentum, weight_decay=1e-5)
    return net, criterion, optimizer

def grad(model):
    # gets the stochastic gradients
    Grad = []
    for p in model.parameters():
        Grad.append(p.grad.clone())
    return Grad

# TRAIN fn
def train(net, criterion, optimizer, loader, epoch, path, batch_verbose=10, iter_stats = [], Grad_Norms = []):
    print('\nEPCH: %d, LR: %0.5f' % (epoch, optimizer.param_groups[0]['lr']))
    print(model_prefix)
    net.train()
    train_loss = 0; correct = 0; total = 0; running_loss = 0.0
    global eval_freq
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        optimizer.zero_grad()
        if "MLP" in args.change_model:
            inputs = inputs.view(inputs.size(0),-1)
        outputs = net(inputs) # obtain logits
        if args.training == "SE":
            # train on SE
            prob = F.softmax(outputs,dim=1)
            top = prob.gather(1,targets.view(-1,1))
            loss = criterion(top, torch.ones(top.shape).cuda())
        elif args.training == "MSE":
            # train on MSE
            prob = F.softmax(outputs,dim=1)
            true_prob = torch.zeros(prob.shape).long().cuda().scatter_(1,targets.view(-1,1),1).float()
            loss = criterion(prob, true_prob)
            loss = torch.mean(torch.sum(loss,dim=1))
        elif args.training == "CE":
            # train on CE
            loss = criterion(outputs, targets)
        loss.backward()
        if args.levy_sigma>0:
            add_noise(net,args.levy_alpha,args.levy_sigma)
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % batch_verbose == batch_verbose-1:
            if args.evaluate_continuous<0:
                print('[ Epoch: %d, Batch #: %5d] Avg-Batch-Verbose-Loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / (batch_verbose) ))
            running_loss = 0.0
        train_loss += (inputs.size(0) * loss.item())
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        update_number = len(loader)*epoch+batch_idx+1
        if args.evaluate_continuous>0 and (update_number<=args.evaluate_continuous or update_number%eval_freq==0):
            train_loss_iter, train_acc_iter, _, _ = evaluate_and_losses(net,criterion,full_train_loader,ps,echo=False)
            test_loss_iter, test_acc_iter, _, _ = evaluate_and_losses(net,criterion,full_test_loader,ps,echo=False)
            iter_stats.append([update_number,train_loss_iter,test_loss_iter,train_acc_iter,test_acc_iter])
            print("#{0:5d}: Train/Test_Loss_{1}: {2:.4f}/{3:.4f} Train/Test_Acc: {4:.3f}/{5:.3f}".format(update_number,args.training,train_loss_iter,test_loss_iter,train_acc_iter,test_acc_iter))
            if len(iter_stats)>10 and np.abs(iter_stats[-1][3]-iter_stats[-2][3])<0.01:
                eval_freq = 50
            if len(iter_stats)>10 and np.abs(iter_stats[-1][3]-iter_stats[-2][3])>=0.01 and np.abs(iter_stats[-1][3]-iter_stats[-2][3])<0.1:
                eval_freq = 10
            if len(iter_stats)>10 and np.abs(iter_stats[-1][3]-iter_stats[-2][3])>=0.1 and np.abs(iter_stats[-1][3]-iter_stats[-2][3])<1:
                eval_freq = 5
            if len(iter_stats)>10 and np.abs(iter_stats[-1][3]-iter_stats[-2][3])>=1:
                eval_freq = 1
        if args.compute_stoch_norm>0 and (update_number%100==0):
            Exact_Grad, Grad_noise = full_grad(net,criterion,online_train_loader)
            Grad_Norms.append((update_number,torch.norm(Grad_noise,dim=1)))
            print("Grad_Norms : Min={} Max={}".format(Grad_Norms[-1][1].min(),Grad_Norms[-1][1].max()))
        if args.save_checkpoint and update_number%1000==0:
            print("Saving update %d" % (update_number))
            save_update(net,update_number,path)
        # print("Batch # %d Loss %0.4f" % (batch_idx+1,loss.item()))
    if args.levy_sigma>0:
        print("Added Levy noise with alpha = {} sigma = {} in all iterations.".format(args.levy_alpha,args.levy_sigma))
    # print("TRN: #PTS: %d LOSS: %0.4f ACC: %0.4f%%" % (total, train_loss/total, (correct * 100.0 / total)))

def full_grad(model,criterion,loader):
    gc.collect()
    device = 'cuda'
    model.eval()
    Grads = []; Sizes = []
    for batch_idx, (inputs, targets) in enumerate(loader):
        optimizer.zero_grad()
        if torch.cuda.is_available():
            inputs, targets = inputs.to(device), targets.to(device)
        if "MLP" in args.change_model:
            inputs = inputs.view(inputs.size(0),-1)
        outputs = model(inputs)
        if args.training == "SE":
            # train on SE
            prob = F.softmax(outputs,dim=1)
            top = prob.gather(1,targets.view(-1,1))
            loss = criterion(top, torch.ones(top.shape).cuda())
        elif args.training == "MSE":
            # train on MSE
            prob = F.softmax(outputs,dim=1)
            true_prob = torch.zeros(prob.shape).long().cuda().scatter_(1,targets.view(-1,1),1).float()
            loss = criterion(prob, true_prob)
            loss = torch.mean(torch.sum(loss,dim=1))
        elif args.training == "CE":
            # train on CE
            loss = criterion(outputs, targets)
        loss.backward()
        grad = [p.grad.cpu().clone() for p in model.parameters()]
        # grad = [p.grad.clone() for p in model.parameters()]
        size = inputs.shape[0]
        Grads.append(grad); Sizes.append(size)
    Flat_Grads = []
    for grad in Grads:
        Flat_Grads.append(torch.cat([g.reshape(-1) for g in grad]))
    Exact_Grad = torch.zeros(Flat_Grads[-1].shape)
    # Exact_Grad = torch.zeros(Flat_Grads[-1].shape).cuda()
    for G,s in zip(Flat_Grads,Sizes):
        Exact_Grad += G*s
    Exact_Grad /= np.sum(Sizes)
    gc.collect()
    Flat_Grads = torch.stack(Flat_Grads)
    Grad_noise = (Flat_Grads-Exact_Grad).cpu()
    # Grad_noise = Flat_Grads-Exact_Grad
    return Exact_Grad, Grad_noise

def estimate_alpha(Z):
    X = Z.reshape(-1); X = X[X.nonzero()]; K = len(X)
    if len(X.shape)>1:
        X = X.squeeze()
    K1 = int(np.floor(np.sqrt(K))); K2 = K1
    X = X[:K1*K2].reshape((K2,K1))
    Y = X.sum(1)
    # X = X.cpu().clone(); Y = Y.cpu().clone()
    a = torch.log(torch.abs(Y)).mean()
    b = (torch.log(torch.abs(X[:K2/4,:])).mean()+torch.log(torch.abs(X[K2/4:K2/2,:])).mean()+torch.log(torch.abs(X[K2/2:3*K2/4,:])).mean()+torch.log(torch.abs(X[3*K2/4:,:])).mean())/4
    return np.log(K1)/(a-b).item()

# EVAL fn

def lp_loss(x,y,p):
    return torch.mean(torch.sum((torch.abs(x-y)**p),dim=1)**(1./p)).item()


def evaluate_and_losses(model,criterion,loader,ps,prob_list=[],features_list=[],store_prob=False,echo=True):
    device = 'cuda'; model.eval()
    test_loss_MSE = 0; test_loss_CE = 0; test_loss_SE = 0; test_loss = 0
    correct = 0; total = 0
    lp_losses = np.zeros(len(ps))
    criterion_CE = nn.CrossEntropyLoss(); criterion_SE = nn.MSELoss(); criterion_MSE = nn.MSELoss(reduction='none')
    Probs = []; Features = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if torch.cuda.is_available():
                inputs, targets = inputs.to(device), targets.to(device)
            if "MLP" in args.change_model:
                inputs = inputs.view(inputs.size(0),-1)
            outputs = model(inputs)
            if args.store_features and batch_idx==0:
                outputs, features = model(inputs,True)
                features_list.append(features.clone())
                del features
            # prob = F.softmax(outputs,dim=1)
            prob = F.softmax(outputs,dim=1)
            true_prob = torch.zeros(prob.shape).long().cuda().scatter_(1,targets.view(-1,1),1).float()
            ########
            if args.evaluate_all or args.training == "MSE":
                loss_MSE = criterion_MSE(prob, true_prob)
                loss_MSE = torch.mean(torch.sum(loss_MSE,dim=1))
                test_loss_MSE += (inputs.size(0) * loss_MSE.item())
            ########
            if args.evaluate_all or args.training == "CE":
                loss_CE = criterion_CE(outputs, targets)
                test_loss_CE += (inputs.size(0) * loss_CE.item())
            ########
            if args.evaluate_all or args.training == "SE":
                top = prob.gather(1,targets.view(-1,1))
                loss_SE = criterion_SE(top, torch.ones(top.shape).cuda())
                test_loss_SE += (inputs.size(0) * loss_SE.item())
            ########
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            ########
            if args.evaluate_all:
                for ip,p in enumerate(ps):
                    lp_losses[ip] += lp_loss(prob,true_prob,p)*inputs.size(0)
            if store_prob:
                Probs += [prob.clone()]
        lp_losses /= total
        if args.training == "MSE":
            test_loss = test_loss_MSE
        elif args.training == "SE":
            test_loss = test_loss_SE
        elif args.training == "CE":
            test_loss = test_loss_CE
        if echo:
            print('TST: #PTS: %d %s_LOSS: %0.4f ACC: %0.4f' % (total, args.training, test_loss/total, (correct * 100.0 / total)))
        if store_prob:
            Probs = torch.cat(Probs,0)
            prob_list += [Probs.clone()]
    model.train()
    if args.evaluate_all:
        return test_loss/total, (correct * 100.0 / total), lp_losses, test_loss_MSE/total, test_loss_CE/total, test_loss_SE/total, prob_list, features_list
    else:
        return test_loss/total, (correct * 100.0 / total), prob_list, features_list

def save_update(model,update_number,path):
    save_dict = {'model' : model.state_dict(), 'update' : update_number}
    full_path = (path + ":{}.iter").format(update_number)
    torch.save(save_dict, full_path)

def save_checkpoint(model, optimizer, epoch, path):
    save_dict = {'model' : model.state_dict(),
                 'optim' : optimizer.state_dict(),
                 'epoch' : epoch }

    full_path = path + ".pt"
    torch.save(save_dict, full_path)

def load_chkpt(model,optimizer,path):
    chkpt = torch.load(path)
    model.load_state_dict(chkpt['model'])
    optimizer.load_state_dict(chkpt['optim'])
    return model, optimizer, chkpt['epoch']

def load_checkpoint(model, optimizer, max_epochs, path):
    found = False
    full_path = ""
    for i in range(max_epochs, -1, -1):
        full_path = (path + ":{}.pt").format(i)
        print("Checking {}".format(full_path))
        if os.path.isfile(full_path):
            print("Checkpoint found at {}".format(full_path))
            try:
                dummy_ckpt = torch.load(full_path); found = True
                break
            except:
                continue
    if not found:
        print("No checkpoint found at {}:N.pt".format(path))
        return model, optimizer, -1
    else:
        checkpoint = torch.load(full_path)
        last_epoch = checkpoint['epoch']
        print("Loading checkpoint for epoch-{} from {}".format(last_epoch, full_path))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        return model, optimizer, last_epoch

def freeze_expect_last_layer(model):
    for p in model.parameters():
        p.requires_grad = False
    list(model.parameters())[-1].requires_grad = True; list(model.parameters())[-2].requires_grad = True

def get_levy_noise(Z,alpha,sigma):
    V = torch.FloatTensor(Z.shape).uniform_(-np.pi/2,np.pi/2).cuda()
    W = torch.FloatTensor(Z.shape).exponential_(1.).cuda()
    X = sigma*(torch.sin(alpha*V)/torch.cos(V)**(1./alpha))*( torch.cos((1-alpha)*V)/W )**((1.-alpha)/alpha)
    return X

def add_noise(model,alpha,sigma):
    for p in model.parameters():
        noise = get_levy_noise(p.grad,alpha,sigma)
        noise[torch.isnan(noise)]=0.
        p.grad += noise

def unflatten_grad(Grad_noise):
    Shapes = []; Grad_Noise = []; start = 0; end = 0
    for p in model.parameters():
        Shapes.append(np.product(p.shape))
    for size,p in zip(Shapes,model.parameters()):
        start = end; end += size; Grad_Noise.append(Grad_noise[:,start:end])
    return Shapes, Grad_Noise

def estimate_block_alpha(Grad_Noise):
    alpha_hat = []
    for g in Grad_Noise:
        alpha_hat.append(estimate_alpha(g))
    return np.array(alpha_hat)

# create transforms
transform_train = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

# create train & test datasets
train_dataset = DATASET(root=data_dir, train=True, download=True, transform=transform_train)
test_dataset = DATASET(root=data_dir, train=False, download=True, transform=transform_test)

if args.random_data:
    train_dataset.train_data = np.random.randint(256,size=train_dataset.train_data.shape,dtype="uint8")
    test_dataset.test_data = np.random.randint(256,size=test_dataset.test_data.shape,dtype="uint8")
    print("Random data feeded")

if args.random_labels:
    train_dataset.train_labels = list(np.random.randint(0,NUM_CLASSES,args.subset))
    print("Assigned random lables to data")

if args.classes!=0:
    train_labels = np.array(train_dataset.train_labels)
    test_labels = np.array(test_dataset.test_labels)
    train_mask = (train_labels<args.classes)
    test_mask = (test_labels<args.classes)
    train_dataset.train_data = train_dataset.train_data[train_mask]
    test_dataset.test_data = test_dataset.test_data[test_mask]
    train_dataset.train_labels = list(train_labels[train_mask])
    test_dataset.test_labels = list(test_labels[test_mask])
    print("Selecting only the first %d classes" % args.classes)
    NUM_CLASSES = args.classes
elif args.classes>NUM_CLASSES:
    print("Please select less than %d classes" % NUM_CLASSES)
    exit()

if args.subset==0:
    args.subset = len(train_dataset)
else:
    print("Selecting a subset of first %d points" % args.subset)

train_dataset.train_data = train_dataset.train_data[:args.subset]
train_dataset.train_labels = train_dataset.train_labels[:args.subset]

num_workers = 0

if args.replacement:
    with_replacement_sampler = torch.utils.data.WeightedRandomSampler(np.ones(NUM_TRAIN_PTS), replacement=True, num_samples=NUM_TRAIN_PTS)
    batch_sampler = torch.utils.data.BatchSampler(with_replacement_sampler, BATCH_SIZE, drop_last=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=num_workers, worker_init_fn=np.random.seed(args.seed))
else:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, worker_init_fn=np.random.seed(args.seed))

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, worker_init_fn=np.random.seed(args.seed))

full_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=FULL_BATCH_SIZE, shuffle=False, num_workers=num_workers, worker_init_fn=np.random.seed(args.seed))
full_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=FULL_BATCH_SIZE, shuffle=False, num_workers=num_workers, worker_init_fn=np.random.seed(args.seed))

if args.compute_stoch_norm>0:
    online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.compute_stoch_norm, shuffle=True, num_workers=num_workers, worker_init_fn=np.random.seed(args.seed))

# trainloader_small = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=num_workers, worker_init_fn=np.random.seed(args.seed))
# testloader_small = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=num_workers, worker_init_fn=np.random.seed(args.seed))

print("# Train Points = %d" % len(train_dataset.train_data)); print("# Test Points = %d" % (len(test_loader.sampler)))

last_epoch = -1
batch_verbose = args.batch_verbose

echo_freq = args.echo_freq
ps = [0.5,1,2,5]
column_names = ['epoch','train_acc','test_acc']
if args.evaluate_all:
    column_names += ['train_loss_MSE','test_loss_MSE']; column_names += ['train_loss_CE','test_loss_CE']; column_names += ['train_loss_SE','test_loss_SE']
    for p in ps:
        column_names += ['train l%0.1f norm'%p]; column_names += ['test l%0.1f norm'%p];
else:
    column_names += ['train_loss_'+args.training, 'test_loss_'+args.training]

if args.evaluate_all:
    cols_to_print = ['train_loss_MSE','train_acc','train_loss_SE','train_loss_CE']
    for p in ps:
        cols_to_print += ['train l%0.1f norm'%p]
else:
    cols_to_print = ['train_loss_'+args.training, 'train_acc']

if args.compute_alpha:
    column_names += ['train_alpha','test_alpha']; cols_to_print += ['train_alpha']

for run_num in range(runs):
    if args.store_prob:
        torch.save(torch.tensor(full_train_loader.dataset.train_labels),CHKPT_PATH+"_train_targets.torch")
        torch.save(torch.tensor(full_test_loader.dataset.test_labels),CHKPT_PATH+"_test_targets.torch")
    model, criterion, optimizer = create_model(MODEL_CLASS,batch_norm=args.batch_norm)
    if args.perturbe_gain>0.:
        model.apply(perturbe_weights)
    if args.evaluate_all:
        stats = {'epoch':[],'train_loss_MSE':[], 'test_loss_MSE':[],'train_acc':[], 'test_acc':[]}
        for p in ps:
            stats['train l%0.1f norm'%p] = []; stats['test l%0.1f norm'%p] = []
        stats['train_loss_CE'] = []; stats['test_loss_CE'] = []; stats['train_loss_SE'] = []; stats['test_loss_SE'] = []
    else:
        stats = {'epoch':[], 'train_loss_'+args.training:[], 'test_loss_'+args.training:[], 'train_acc':[], 'test_acc':[]}
    if args.compute_alpha:
        stats['train_alpha'] = []; stats['test_alpha'] = [];
    df = pd.DataFrame(stats); df = df[column_names]
    start_epoch = last_epoch + 1; i = start_epoch; j = 0; iteration = 0; iterations = []
    train_prob_list = []; test_prob_list = []; features_list = []
    # alpha_hats = [];
    Grad_Norms = []; iter_stats = []
    if args.freeze:
        MAX_EPOCHS += args.freeze
    eval_freq = 10
    for i in range(start_epoch, MAX_EPOCHS):
        path = CHKPT_PATH+"_"+str(run_num)
        if i==args.freeze:
            freeze_expect_last_layer(model)
            print("Freezing all layers expect the last layer\n\n")
        train(model, criterion, optimizer, train_loader, i, path, batch_verbose, iter_stats = iter_stats, Grad_Norms=Grad_Norms)
        # evaluate on train and test set
        if i%echo_freq==0 or i<30:
            if args.evaluate_all:
                train_loss, train_acc, train_lp_losses, train_loss_MSE, train_loss_CE, train_loss_SE, train_prob_list, features_list = evaluate_and_losses(model,criterion,full_train_loader,ps,prob_list=train_prob_list,features_list=features_list,store_prob=args.store_prob)
                test_loss, test_acc, test_lp_losses, test_loss_MSE, test_loss_CE, test_loss_SE, test_prob_list, _ = evaluate_and_losses(model,criterion,full_test_loader,ps,prob_list=test_prob_list,store_prob=args.store_prob)
            else:
                train_loss, train_acc, train_prob_list, features_list = evaluate_and_losses(model,criterion,full_train_loader,ps,prob_list=train_prob_list,features_list=features_list,store_prob=args.store_prob)
                test_loss, test_acc, test_prob_list, _ = evaluate_and_losses(model,criterion,full_test_loader,ps,prob_list=test_prob_list,store_prob=args.store_prob)
            if args.compute_alpha:
                Exact_Grad, Grad_noise = full_grad(model,criterion,online_train_loader); train_alpha = estimate_alpha(Grad_noise)
                Exact_Grad, Grad_noise = full_grad(model,criterion,test_loader); test_alpha = estimate_alpha(Grad_noise)
            stats_new = {'epoch': [i+1],'train_acc': [train_acc], 'test_acc': [test_acc]}
            if args.evaluate_all:
                stats_new['train_loss_MSE'] = [train_loss_MSE]; stats_new['test_loss_MSE'] = [test_loss_MSE]; stats_new['train_loss_CE'] = [train_loss_CE]; stats_new['test_loss_CE'] = [test_loss_CE]; stats_new['train_loss_SE'] = [train_loss_SE]; stats_new['test_loss_SE'] = [test_loss_SE]
                for ip,p in enumerate(ps):
                    stats_new['train l%0.1f norm'%p] = [train_lp_losses[ip]]; stats_new['test l%0.1f norm'%p] = [test_lp_losses[ip]]
            else:
                stats_new['train_loss_'+args.training] = [train_loss]; stats_new['test_loss_'+args.training] = [test_loss]
            if args.compute_alpha:
                stats_new['train_alpha'] = [train_alpha]; stats_new['test_alpha'] = [test_alpha];
            df_new = pd.DataFrame(stats_new); df_new = df_new[column_names]
            df = pd.concat((df, df_new)); df = df.reset_index(drop=True); df = df[column_names]
            df.to_csv(CHKPT_PATH+"_stats.csv",index=False)
            print(df.iloc[-1][cols_to_print])
        if args.stop_when_trained and len(df)>10:
            if df.iloc[-5]['train_acc']>99.999:
                break
        gc.collect()
    if args.compute_stoch_norm>0:
        Grad_Norms_iter, Grad_Norms = zip(*Grad_Norms); Grad_Norms_iter = list(Grad_Norms_iter); Grad_Norms = list(Grad_Norms)
        Grad_Norms = torch.stack(Grad_Norms)
        d = {"iterations":Grad_Norms_iter, "Grad_Norms":Grad_Norms}
        torch.save(d,CHKPT_PATH+"_grad_norms.torch")
    if args.store_prob:
        train_prob_array = torch.stack(train_prob_list); test_prob_array = torch.stack(test_prob_list);
        torch.save(train_prob_array,CHKPT_PATH+"_train_prob.torch"); torch.save(test_prob_array,CHKPT_PATH+"_test_prob.torch")
    if args.store_features:
        feature_array = torch.stack(features_list); torch.save(feature_array,CHKPT_PATH+"_features.torch")
    print("Run number %d done" % (run_num))
    if args.evaluate_all:
        test_loss, test_acc, test_lp_losses, test_loss_MSE, test_loss_CE, test_loss_SE, _, _ = evaluate_and_losses(model,criterion,full_test_loader,ps)
    else:
        test_loss, test_acc, _, _ = evaluate_and_losses(model,criterion,full_test_loader,ps)
    if args.evaluate_continuous>0:
        df_iter = pd.DataFrame(iter_stats)
        df_iter.columns = ['iterations','train_loss','test_loss','train_acc','test_acc']
        df_iter.to_csv(CHKPT_PATH+"_iter_stats.csv",index=False)
print("Exiting")

# run these commands on vim on philly after :
# %s/train_dataset.train_data/train_dataset.data/g
# %s/test_dataset.test_data/test_dataset.data/g
# %s/train_dataset.train_labels/train_dataset.targets/g
# %s/test_dataset.test_labels/test_dataset.targets/g
# %s/full_train_loader.dataset.train_labels/full_train_loader.dataset.targets/g
# %s/train_dataset.train_data/train_dataset.data/g | %s/test_dataset.test_data/test_dataset.data/g | %s/train_dataset.train_labels/train_dataset.targets/g | %s/test_dataset.test_labels/test_dataset.targets/g | %s/full_train_loader.dataset.train_labels/full_train_loader.dataset.targets/g
