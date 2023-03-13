#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import shutil
import os
RESULTS_FILE_PATH = '../data/cats_dogs/test/'

# move all files to labeled folders dog = 1 , cat= 2
filelist = [ f for f in os.listdir(RESULTS_FILE_PATH) if  f.endswith(".jpg") ]
for file in filelist:
    print file
    label = file.split('.')[0].replace('dog','1').replace('cat','2')
    if not os.path.isdir(RESULTS_FILE_PATH+label):
        print('new directry has been created')
        os.system('mkdir ' + RESULTS_FILE_PATH+label)
    shutil.move( RESULTS_FILE_PATH + file, RESULTS_FILE_PATH + label+'/'+file)





import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# Hyper Parameters
num_epochs = 1000
batch_size = 100
learning_rate = 0.01
hidden_size = 20


# best model for saving
best_model = 0.1
# Image Preprocessing 
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.247, 0.2434, 0.2615)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.247, 0.2434, 0.2615)),
])

# define a model:
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,16 , kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 20, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(20, 20, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2))
        self.fc1 = nn.Linear(4*4*20, 3)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax()
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return self.logsoftmax(out)
    



def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

cnn = CNN()
if torch.cuda.is_available():
    cnn = cnn.cuda()
# Hyper Parameters
num_epochs = 1000
batch_size = 100
learning_rate = 0.01
hidden_size = 28

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Scale((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.247, 0.2434, 0.2615)),
])
transform_test = transforms.Compose([
    transforms.Scale((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.247, 0.2434, 0.2615)),
])

DATASET_DIR = './data/cats_dogs/'
dataset_test = torchvision.datasets.ImageFolder(root=DATASET_DIR, transform=transform_test)

DATASET_DIR = './data/cats_dogs/'
dataset = torchvision.datasets.ImageFolder(root=DATASET_DIR, transform=transform_train)


train_loader = torch.utils.data.DataLoader(dataset=dataset_test,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

test_loader.dataset.classes

cnn = CNN()
if torch.cuda.is_available():
    cnn = cnn.cuda()

# convert all the weights tensors to cuda()
# Loss and Optimizer

criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate, momentum=0.9)

print ('number of parameters: ', sum(param.numel() for param in cnn.parameters()))
report_str = 'number of parameters: ', sum(param.numel() for param in cnn.parameters())
# training the model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = to_var(images)
        labels = to_var(labels)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 1 == 10:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1,
                     len(train_dataset)//batch_size, loss.data[0]))


# evaluating the model
    cnn.eval()
    correct = 0
    total = 0
    current_model = 0
    for images, labels in test_loader:
        images = to_var(images)
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
        current_model = (1 - float(correct) / total)
    print ('Test Error of the model on the 10000 test images: %.4f' % current_model)
    print ('Best so far :%.4f' % best_model)
    

