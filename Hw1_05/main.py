from pickle import TRUE
from turtle import screensize, title
from PyQt5 import QtWidgets,QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import os
import numpy as np
import matplotlib.pyplot as plt
import sys, UI
import cv2
from torchsummary import summary
import tkinter as tk
import argparse
import torchvision
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.transforms as T
import matplotlib.image as mpimg

images=[]


transform = transforms.Compose(
    [transforms.ToTensor(),
     ])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=9,
                                          shuffle=True, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')




# get some random training images





class myMainWindow(QMainWindow, UI.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.Ima)
        self.pushButton_2.clicked.connect(self.ST)
        self.pushButton_3.clicked.connect(self.MS)
        self.pushButton_4.clicked.connect(self.DA)
        self.pushButton_5.clicked.connect(self.SD)
        self.pushButton_6.clicked.connect(self.PD)
        
    def Ima(self):
        images.clear()
        filename,_=QFileDialog().getOpenFileName(self,"Load am")
        if(filename==''): return
        print(filename)
        img = Image.open(filename)
        scene = QtWidgets.QGraphicsScene()    
        scene.setSceneRect(0, 0, 445, 445)
        imgs = QtGui.QPixmap(filename)        
        imgs = imgs.scaled(445,445)  
        scene.addPixmap(imgs)                 
        self.graphicsView.setScene(scene) 
        images.append(img)
    def ST(self):
        dataiter = iter(trainloader)
        images, labels = next(dataiter)
        num_plot = 3
        f, ax = plt.subplots(num_plot, num_plot)
        for m in range(9):
            ax[int(m/3), m%3].imshow(images[m].permute(1, 2, 0))
            ax[int(m/3), m%3].set_title(str(classes[labels[m]]),fontsize=8)
            ax[int(m/3), m%3].get_xaxis().set_visible(False)
            ax[int(m/3), m%3].get_yaxis().set_visible(False)
        f.set_size_inches(5.5, 5.5)
        f.subplots_adjust(hspace=0.2)
        f.subplots_adjust(wspace=0)
        plt.show()      
    def MS(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('GPU state:', device)
        model = torch.load("model.pth")
        model = model.to(device)
        summary(model, input_size=(3,224,224)) 
    def DA(self):
        if(len(images)==0): return
        temp=images[0]
        rotater = T.RandomRotation(degrees=(-30, 30))
        rotated_imgs = [rotater(temp) for _ in range(4)]
        resize_cropper = T.RandomResizedCrop(size=(36, 36))
        resized_crops = [resize_cropper(temp) for _ in range(4)]
        hflipper = T.RandomHorizontalFlip(p=1)
        flipimg = [hflipper(temp) for _ in range(4)]
        f, ax = plt.subplots(1,3)
        ax[0].imshow(rotated_imgs[0])
        ax[0].set_title('RandomRotation',fontsize=8)
        ax[0].get_xaxis().set_visible(False)
        ax[0].get_yaxis().set_visible(False)
        ax[1].imshow(resized_crops[0])
        ax[1].set_title('RandomResizedCrop',fontsize=8)
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)
        ax[2].imshow(flipimg[0])
        ax[2].set_title('RandomHorizontalFlip',fontsize=8)
        ax[2].get_xaxis().set_visible(False)
        ax[2].get_yaxis().set_visible(False)
        f.set_size_inches(5.5, 5.5)
        f.subplots_adjust(hspace=0)
        f.subplots_adjust(wspace=0.2)
        plt.show()
    def SD(self):
        scene = QtWidgets.QGraphicsScene()
        scene.setSceneRect(0, 0, 445, 445)
        imgs = QtGui.QPixmap('Draw.png')     
        imgs = imgs.scaled(445,445)  
        scene.addPixmap(imgs)                    
        self.graphicsView.setScene(scene) 
    def PD(self):
        if(len(images)==0): return
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('GPU state:', device)
        model = torch.load("model.pth")
        model = model.to(device)
        model.eval()
        transform = transforms.Compose([transforms.Resize(size=(224, 224)),transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) )])
        img = images[0]
        TRima = transform(img).unsqueeze(0)
        TRima = TRima.to(device)
        out = model(TRima)
        _ , renum = torch.max(out,1)
        confidence = torch.nn.functional.softmax(out, dim=1)[0]
        fconfidence = confidence[int(renum)].item()
        result = classes[renum]
        prt='Confidence = %f'%(fconfidence)
        resultt='Prediction Label = '+result
        self.label.setText(prt)
        self.label_2.setText(resultt)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = myMainWindow()
    window.show()
    sys.exit(app.exec_())