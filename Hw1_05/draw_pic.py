from audioop import bias
import random
from turtle import screensize, title
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import os
import numpy as np
import matplotlib.pyplot as plt
import sys, UI
import cv2
import re
import tkinter as tk
import argparse
import pickle
import torch.nn as nn
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
import torchvision.transforms as transforms
data=[86,93.75]
# import matplotlib相關套件

import matplotlib.pyplot as plt

# import字型管理套件

from matplotlib.font_manager import FontProperties

 

# 指定使用字型和大小

# accuracy loss

 
data = [2.5,0.4145,0.22372,0.224284,0.224774,0.207682,0.189686,0.117436,0.126328,0.129302,0.08842,0.07879,0.068508,0.052524,0.055552,0.064838,0.04547,0.045318,0.020594,0.0165398,0.031276,0.029234,0.020456,0.018056,0.008258,0.00924,0.01902,0.025962,0.033466,0.013634,0.008818]
data1=[0,86,93.75,92.5,92.75,91.5,95,96.25,96.25,94.5,97.75,97.5,97.75,98.25,98.75,98,98,97.75,99.5,99.25,99,98.75,99.25,99.25,99.75,99.75,99.25,99.5,99.5,99.25,99.75]

epoch = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
plt.figure(figsize=(8,4),linewidth = 2)
plt.plot(epoch,data,color = 'r', label="Training loss")
plt.title("Loss",  x=0.5, y=1.03,fontsize=14)
plt.legend(loc='upper right')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.show()