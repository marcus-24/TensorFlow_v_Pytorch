import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torcheval.metrics import MulticlassAccuracy, Mean

from customtorchutils import get_workers

BATCH_SIZE = 32
N_CLASSES = 10
EPOCHS = 5

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same')
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=256 * 2 *2, out_features=512) # max pool dimensions times conv2s output
        self.fc2 = nn.Linear(in_features=512, out_features=N_CLASSES)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Defines forward pass of data through network'''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # do not need softmax since done for you in CrossEntropyLoss
        return x
    
    def compile(self, 
                optimizer, 
                criterion,
                metrics,
                loss_compiler) -> None:
        self._optimizer = optimizer
        self._criterion = criterion
        self._metrics = metrics
        self._loss_compiler = loss_compiler

    def fit(self,
            train_data: DataLoader,
            val_data: DataLoader,
            epochs: int,
            device: str):
        for epoch in range(epochs):  # loop over the dataset multiple times

            self.train()
            for data in tqdm(train_data, desc=f'Epoch: {epoch + 1}/{epochs}', total=len(train_data)):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                self._optimizer.zero_grad() # resets the gradients for new batch

                # forward + backward + optimize
                outputs = self._forward(inputs) # predicted output
                loss = self._criterion(outputs, labels) # calulate loss for batch
                loss.backward() # perform backprogation to calculate gradients
                self._optimizer.step() # gradient descent - update network weights and biases

                # print statistics
                _, predicted = torch.max(outputs.data, 1)
                self._metrics.update(predicted, labels)
                self._loss_compiler.update(loss)

            print(f'Average loss={self._loss_compiler.compute():.4f} \
                   Average accuracy={self._metrics.compute() * 100:.3f}%', end='\t')

            self._loss_compiler.reset()
            self._metrics.reset()

            self.eval()
            for data in val_data:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self._forward(images)
                _, predicted = torch.max(outputs.data, 1)
                self._metrics.update(predicted, labels)
                self._loss_compiler.update(loss)

            print(f'Val. Average loss={self._loss_compiler.compute():.4f} \
                  Val. Average accuracy={self._metrics.compute() * 100:.3f}%')
            
            self._loss_compiler.reset()
            self._metrics.reset()

            
        
if __name__ == "__main__":
    device, num_workers = get_workers()

    '''Create a pipeline of transformations'''
    training_transform = transforms.Compose(
        [transforms.ToTensor(), # converts image to tensor. Also scales image to [0,1]
        transforms.RandomHorizontalFlip(p=0.5)])

    test_transform = transforms.ToTensor()

    trainset = torchvision.datasets.CIFAR10(root='./data', 
                                            train=True,
                                            download=True, 
                                            transform=training_transform)
    trainloader = DataLoader(trainset,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             num_workers=num_workers,
                             generator=torch.Generator(device=device))

    testset = torchvision.datasets.CIFAR10(root='./data', 
                                           train=False,
                                           download=True, 
                                           transform=test_transform)
    testloader = DataLoader(testset,
                            batch_size=BATCH_SIZE,
                            shuffle=False, 
                            num_workers=num_workers,
                            generator=torch.Generator(device=device))

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    metrics = MulticlassAccuracy(device=device)

    net.compile(optimizer=optimizer,
                criterion=criterion,
                metrics=metrics,
                loss_compiler=Mean())
    
    net.fit(train_data=trainloader,
            val_data=testloader,
            epochs=EPOCHS,
            device=device)
