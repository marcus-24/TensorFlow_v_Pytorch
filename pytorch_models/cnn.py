import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 0 if device == "cuda" else 2  # avoids runtime error since gpu cant have multiple workers
    print('device used: ', device)
    print('num workers: ', num_workers)
    torch.set_default_device(device)

    '''Create a pipeline of transformations'''
    training_transform = transforms.Compose(
        [transforms.ToTensor(), # converts image to tensor. Also scales image to [0,1]
        transforms.RandomHorizontalFlip(p=0.5)])

    test_transform = transforms.ToTensor()

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=training_transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            num_workers=num_workers,
                                            generator=torch.Generator(device=device))

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=num_workers,
                                            generator=torch.Generator(device=device))

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    n_train_batches = len(trainloader)
    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        running_loss = 0.0
        correct = 0
        total_samples = 0
        for i, data in tqdm(enumerate(trainloader), desc=f'Epoch: {epoch + 1}/{EPOCHS}', total=n_train_batches):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad() # resets the gradients for new batch

            # forward + backward + optimize
            outputs = net(inputs) # predicted output
            loss = criterion(outputs, labels) # calulate loss for batch
            loss.backward() # perform backprogation to calculate gradients
            optimizer.step() # gradient descent - update network weights and biases

            # print statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = running_loss / n_train_batches
        avg_acc = 100 * correct / total_samples  # TODO: Double check math
        print(f'Average loss={avg_loss} \t Average accuracy={avg_acc}%')


    print('Finished Training')


    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')