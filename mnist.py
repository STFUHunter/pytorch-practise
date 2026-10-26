import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#create transform X
#get data X
#initiate dataloaders (train, test) X
#create class model def init forward X
#initiate model, criterion, optimizer X
#training loop X
#eval loop X
#export weights X
#display metrics X

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = torchvision.datasets.MNIST(root='./data2', train=True, transform=transform, download=True)
test_data = torchvision.datasets.MNIST(root='./data2', train=False, transform=transform, download=True)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 3 * 3, 512)
        self.dorpout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(512, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = x.view(-1, 64 * 3 * 3)

        x = self.relu(self.fc1(x))
        x = self.dorpout(x)
        x = self.fc2(x)

        return x
if "__name__" == "__main__":
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    metrics = {
        'loss': [],
        'lr': []
    }
    epochs = 20

    print('Model is training')

    for epoch in range (epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        metrics['loss'].append(epoch_loss)
        metrics['lr'].append(current_lr)

        print(f'Epoch {epoch+1}, loss {running_loss / len(train_loader):.4f}')

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        print('Model is evaluating')
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')

    torch.save(model.state_dict(), "mnist_cnn.pth")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(metrics['loss'], label='Training-loss', color='crimson', linewidth=2)
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(metrics["lr"], label='Learning Rate', color='darkblue', linewidth=2)
    plt.title('Optimizer Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()