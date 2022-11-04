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
traffic=[]
am=[]
images=[]
filenames=[]



class myMainWindow(QMainWindow, UI.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.Load_traff)
        self.pushButton_2.clicked.connect(self.Load_Am)
        self.pushButton_3.clicked.connect(self.KP)
        self.pushButton_4.clicked.connect(self.KPM)
        
        
    def KPM(self):
        traffico=traffic[0].copy()
        gray= cv2.cvtColor(traffico,cv2.COLOR_BGR2GRAY)
        sift2 = cv2.xfeatures2d.SIFT_create()
        kp = sift2.detect(gray,None)
        key=cv2.drawKeypoints(gray,kp,traffico,(0,255,0))

        sift = cv2.xfeatures2d.SIFT_create()
        img1=cv2.cvtColor(traffic[0],cv2.COLOR_BGR2GRAY)
        img2=cv2.cvtColor(am[0],cv2.COLOR_BGR2GRAY)
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)

        good = []
        for m,n in matches:
            if m.distance < 0.685*n.distance:
                good.append([m])
        img3 = cv2.drawMatchesKnn(key,kp1,img2,kp2,good,None,(0,255,255),flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('Figure 2 ',img3)
        cv2.imwrite('Figure 2.png',img3)
    def KP(self):
        traffico=traffic[0].copy()
        gray= cv2.cvtColor(traffico,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(gray,None)
        key=cv2.drawKeypoints(gray,kp,traffico,(0,255,0))
        cv2.imwrite('Figure 1.png',key)
        cv2.imshow('Figure 1 ',key)
    def Load_traff(self):
        traffic.clear()
        filename,_=QFileDialog().getOpenFileName(self,"Load traffic")
        if(filename==''): return
        temp= cv2.imread(filename)
        traffic.append(temp)
        
    def Load_Am(self):
        am.clear()
        filename,_=QFileDialog().getOpenFileName(self,"Load am")
        if(filename==''): return
        temp= cv2.imread(filename)
        am.append(temp)

         


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = myMainWindow()
    window.show()
    sys.exit(app.exec_())