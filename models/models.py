import torch
from torch import nn
from utils import pt_util


class AlexNet(nn.Module):
    def __init__(self, input_height=32, input_width=32, input_channels=3, ch=64, output_size=10,
                 criterion=nn.CrossEntropyLoss()):
        super(AlexNet, self).__init__()
        self.criterion = criterion
        self.best_accuracy = 0

        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.output_size = output_size

        self.features = nn.Sequential(
            nn.Conv2d(3, out_channels=ch, kernel_size=4, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(ch, ch, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, self.output_size)
        )

    def forward(self, inputs):
        outputs = self.features(inputs)
        outputs = outputs.squeeze()
        outputs = self.classifier(outputs)
        return outputs

    def loss(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        return loss

    def save_model(self, file_path, num_to_keep=1):
        pt_util.save(self, file_path, num_to_keep)

    def save_best_model(self, accuracy, file_path, num_to_keep=1):
        # TODO save the model if it is the best
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.save_model(file_path, num_to_keep)

    def load_model(self, file_path):
        pt_util.restore(self, file_path)

    def load_last_model(self, dir_path):
        return pt_util.restore_latest(self, dir_path)


class MLP(nn.Module):
    def __init__(self, input_size=32*32*3, hidden_size=1024, output_size=10, num_hidden_layers=2,
                 criterion=nn.CrossEntropyLoss(), batch_norm=True):
        super(MLP, self).__init__()
        self.criterion = criterion
        self.best_accuracy = 0

        self.num_hidden_layers = num_hidden_layers
        if num_hidden_layers == 0:
            self.classifier = nn.Linear(input_size, output_size)
        else:
            self.fc = [nn.Linear(input_size, hidden_size)]
            if batch_norm:
                self.fc.append(nn.BatchNorm1d(hidden_size))
            self.fc.append(nn.ReLU())
            for i in range(1, num_hidden_layers):
                self.fc.append(nn.Linear(hidden_size, hidden_size))
                if batch_norm:
                    self.fc.append(nn.BatchNorm1d(hidden_size))
                self.fc.append(nn.ReLU())
            self.layers = nn.Sequential(*self.fc)
            self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        if self.num_hidden_layers == 0:
            return self.classifier(inputs)
        outputs = self.layers(inputs)
        outputs = self.classifier(outputs)
        return outputs

    def loss(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        return loss

    def save_model(self, file_path, num_to_keep=1):
        pt_util.save(self, file_path, num_to_keep)

    def save_best_model(self, accuracy, file_path, num_to_keep=1):
        # TODO save the model if it is the best
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.save_model(file_path, num_to_keep)

    def load_model(self, file_path):
        pt_util.restore(self, file_path)

    def load_last_model(self, dir_path):
        return pt_util.restore_latest(self, dir_path)


def test_mlp():
    model = MLP(input_size=784, hidden_size=256, num_hidden_layers=1, output_size=10)
    inputs = torch.randn(1000, 784)
    print("inputs size", inputs.size())
    outputs = model(inputs)
    print("outputs size", outputs.size())
    targets = torch.randint(low=0, high=10, size=(1000,))
    loss = model.loss(outputs, targets)
    print("loss value", loss)
    params = model.state_dict().keys()
    print(params, '\n')


def test_alexnet():
    model = AlexNet()
    # Imagine 50 sample images
    inputs = torch.randn(50, 3, 32, 32)
    print("inputs size", inputs.size())
    outputs = model(inputs)
    print("outputs size", outputs.size())
    targets = torch.randint(low=0, high=10, size=(50,))
    loss = model.loss(outputs, targets)
    print("loss value", loss)
    params = model.state_dict().keys()
    print(params, '\n')


if __name__ == '__main__':
    test_mlp()
    test_alexnet()
