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
from PIL import Image
import matplotlib.pyplot as plt


#transforms data processoing
transformer= transforms.Compose({
    transforms.Resize((150,150)), #Making same size
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5, 0.5, 0.5], #0-1 to [-1,1] , formula (x-mean)/std
                          [0.5, 0.5, 0.5])
})

# Path for training and testing directory
train_path = r"C:\Users\ASUS\Downloads\Project detection\seg_train\seg_train"
pred_path = r"C:\Users\ASUS\Downloads\Project detection\seg_pred\seg_pred"

#catgories
root=pathlib.Path(train_path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])

class ConvNet(nn.Module):
    def __init__(self, num_classes=6):
        super(ConvNet, self).__init__()

        # Creating first Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Creating second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        # Creating third Convolutional Layer
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
        output = output.reshape(output.size(0), -1)  # Reshaping the matrix size using view function

        output = self.fc(output)  # Fit it inside the fully-connected layer

        return output

checkpoint=torch.load('best_checkpoint.model')
model=ConvNet(num_classes=6)
model.load_state_dict(checkpoint)
model.eval()

transformer2= transforms.Compose({
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Resize((150,150)), #Making same size
    transforms.ToTensor(), #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5, 0.5, 0.5], #0-1 to [-1,1] , formula (x-mean)/std
                          [0.5, 0.5, 0.5]),
    
})

class_names = ['building', 'forest', 'glacier', 'mountain', 'sea', 'street']
#Prediction Function
def predict(image_path):

    x = plt.imread(image_path)
    img = Image.fromarray(x)
    img = img.resize((150,150))
    x = np.asarray(img)

    x = x.reshape(1, 150, 150, 3)
    x = x.transpose((0, 3, 1, 2))  # necessary because PyTorch requires the array in different order
    x = x.astype(np.float32)      
    x /= 255.0  # to scale pixel values to probability format              

    x_tensor = torch.from_numpy(x) # from numpy array to tensor
    # print(x_tensor.shape)
    output = model(x_tensor)
    index = output.data.cpu().numpy().argmax()  #index is number of predicted class
    predicted_class = class_names[index]
    # Here I should have classes list similar to ['mountain', ...] so I can see the class by printing class[index]
    # then I return index or class name instead of raw output
    img = plt.imread(image_path)
    # Displaying the image
    # plt.imshow(img)
    # plt.show()

    return predicted_class
 #os.listdir 
# for item in os.listdir(r"images\\"):
# print(predict("fileName"))
#print(os.listdir(r"images\\"))