
import torch
import cv2 
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as trans
import time
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from matplotlib import cm
import torchvision.transforms as transforms
classes = ('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load("./123/model.pth")
model = model.to(device)
model.eval()

img_path = '1596705248-5f2bc9e08a332.jpg'
        
transform = transforms.Compose([transforms.Resize(size=(224, 224)),transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

img = Image.open(img_path)
imgT = transform(img).unsqueeze(0)

imgT = imgT.to(device)
outputs = model(imgT)

_ , indices = torch.max(outputs,1)
percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
confidenceO = torch.nn.functional.softmax(outputs, dim=1)[0]
perc = percentage[int(indices)].item()
confidence = confidenceO[int(indices)].item()
result = classes[indices]
print('predicted:', result, perc)

showimg = mpimg.imread(img_path)
imgplot = plt.imshow(showimg)
plt.title("Confidence = %f \n Prediciton Label: %s" %(confidence, result))
plt.tight_layout(True)
plt.show()