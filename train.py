#Loading Libraries
import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib
import torchvision.transforms as transforms
#checking for device
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformer = transforms.Compose([
    transforms.Resize((150, 150)),  # Making same size
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(30),  # Random rotation within 30 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random brightness, contrast, saturation, and hue adjustments
    transforms.ToTensor(),  # 0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5, 0.5, 0.5],  # 0-1 to [-1,1], formula (x-mean)/std
                         [0.5, 0.5, 0.5])
])

transformer2 = torchvision.transforms.Compose([
<<<<<<< HEAD
    # Add your transformations here, for example:
=======
    # Adding transformations
>>>>>>> 9ce17a3 (Updated)
    torchvision.transforms.Resize((150, 150)),
    torchvision.transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], #0-1 to [-1,1] , formula (x-mean)/std
                          [0.5, 0.5, 0.5])
])

#Dataloader helps to read data

<<<<<<< HEAD
# Path for training and testing directory
train_path = r"C:\Users\ASUS\Downloads\Project detection\seg_train\seg_train"
test_path = r"C:\Users\ASUS\Downloads\Project detection\seg_test\seg_test"

# transformer = torchvision.transforms.Compose([
#     # Add your transformations here, for example:
#     torchvision.transforms.Resize((150, 150)),
#     torchvision.transforms.ToTensor()
# ])

=======
# Here, I added Path for training and testing directory
train_path = r"C:\Users\ASUS\Downloads\Project detection\seg_train\seg_train"
test_path = r"C:\Users\ASUS\Downloads\Project detection\seg_test\seg_test"
#Here I am Training DataLoader
>>>>>>> 9ce17a3 (Updated)
train_loader = DataLoader(
    torchvision.datasets.ImageFolder(train_path, transform=transformer),
    batch_size=16, shuffle=True
)

test_loader = DataLoader(
    torchvision.datasets.ImageFolder(test_path, transform=transformer2),
    batch_size=1, shuffle=True
)

#catgories
root=pathlib.Path(train_path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])

print(classes)

class ConvNet(nn.Module):
    def __init__(self, num_classes=6):
        super(ConvNet, self).__init__()

<<<<<<< HEAD
        # Creating first Convolutional Layer
=======
        # Here I am creating first Convolutional Layer
>>>>>>> 9ce17a3 (Updated)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

<<<<<<< HEAD
        # Creating second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        # Creating third Convolutional Layer
        #
=======
        # Here I am creating second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        # Here I am creating third Convolutional Layer
>>>>>>> 9ce17a3 (Updated)
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()

        # Fully connected layer
        self.fc = nn.Linear(in_features=180000, out_features=num_classes)

    def forward(self, input2):

        output = self.conv1(input2)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.pool(output)


        output = self.conv2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        
        # Above output will be in the matrix form, with shape (256, 32, 75, 75)
        output = output.view(output.size(0), -1)  # Reshaping the matrix size using view function

        output = self.fc(output)  # Fit it inside the fully-connected layer

        return output
    
model=ConvNet(num_classes=6).to(device)#Initializing the model and moves it to device
#Optimizer and loss function

optimizer=Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
<<<<<<< HEAD
loss_function=nn.CrossEntropyLoss()# Measure the precision of model prediction vs actual labels
=======
loss_function=nn.CrossEntropyLoss()# Here it measure the precision of model prediction vs actual labels
>>>>>>> 9ce17a3 (Updated)

num_epochs=12 #1o itirations 

#calculating the size of training and testing images
train_count=len(glob.glob(train_path+'/**/*.jpg'))
test_count=len(glob.glob(test_path+'/**/*.jpg'))

print(train_count,test_count)

best_accuracy = 0.0
for epoch in range(num_epochs):

    # Training on training dataset
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, prediction = torch.max(outputs.data, 1)
        train_accuracy += int(torch.sum(prediction == labels.data))

    train_accuracy = train_accuracy / train_count
    train_loss = train_loss / train_count

    # Evaluation on testing dataset
    model.eval()
    test_accuracy = 0.0

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, prediction = torch.max(outputs.data, 1)
            test_accuracy += int(torch.sum(prediction == labels.data))

    test_accuracy = test_accuracy / test_count

    print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

<<<<<<< HEAD
    # Save the best model
=======
    # Here I am saving the best model
>>>>>>> 9ce17a3 (Updated)
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), 'best_checkpoint.model')
        best_accuracy = test_accuracy

