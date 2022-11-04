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

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

objp = np.zeros((8*11,3), np.float32)
objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
objpoints = []
imgpoints = []    
InputW=""
imgLL=[]
imgRL=[]
images=[]
filenames=[]
alert1="****PRESS Q TO EXIT DRWING****"
alert="*****PLEASE DON'T CLOSE THIS WINDOW!!*****"
def cacu():
    objpoints.clear()
    imgpoints.clear()
    for img in images:
        gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        ret, corner=cv2.findChessboardCorners(gray, (11,8),None)
        if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corner, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2)

def load_images_from_folder(folder):
    images.clear()
    filenames.clear()
    for filename in sorted_alphanumeric(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            filenames.append(filename)
            images.append(img)

def draw1(image,imgpts):
    image = cv2.line(image, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0, 0, 255), 5)
    return image

class myMainWindow(QMainWindow, UI.Ui_MainWindow):
    def __init__(self):
         super().__init__()
         self.setupUi(self)
         self.pushButton.clicked.connect(self.Load_Image)
         self.pushButton_11.clicked.connect(self.Find_corner)
         self.pushButton_10.clicked.connect(self.Find_IN)
         self.pushButton_12.clicked.connect(self.Find_EI)
         self.pushButton_13.clicked.connect(self.Find_DI)
         self.pushButton_14.clicked.connect(self.Undis)
         self.pushButton_15.clicked.connect(self.SWB)
         self.pushButton_18.clicked.connect(self.SWV)
         self.pushButton_2.clicked.connect(self.LF)
         self.pushButton_3.clicked.connect(self.LR)
         self.pushButton_19.clicked.connect(self.STM)
    def STM(self):
        imgL=imgLL[0]
        imgR=imgRL[0]
        tempL=cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
        tempR=cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
        stereo = cv2.StereoBM_create(256,25)
        disparity = stereo.compute(tempL, tempR).astype(np.float32)/16.0
        m = np.ones(disparity.shape, dtype="float32")
        disparity = cv2.add(disparity, m)
        disparity=disparity.astype(np.uint8)
        disparityshow= cv2.resize(disparity , (700, 425), interpolation=cv2.INTER_AREA)
        cv2.namedWindow('disparity '+alert)
        cv2.resizeWindow('disparity '+alert,700,425)
        cv2.imshow('disparity '+alert, disparityshow)
        imgL=cv2.resize(imgLL[0] , (700, 425), interpolation=cv2.INTER_AREA)
        imgR=cv2.resize(imgRL[0] , (700, 425), interpolation=cv2.INTER_AREA)
        tempL=cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
        tempR=cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
        stereo = cv2.StereoBM_create(128,13)
        disparity = stereo.compute(tempL, tempR)/16
        global tempRR
        tempRR=imgR.copy()
        global tempLL
        tempLL=imgL.copy()
        def draw_circle(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                global tempRR
                tempRR=imgR.copy()
                global tempLL
                tempLL=imgL.copy()
                cv2.circle(tempLL,(x,y),2,(0,255,0),2)
                cv2.circle(tempRR,(int(x-disparity[y][x]),y),2,(0,255,0),2)
        while(1):
            cv2.namedWindow('imageL '+alert1)
            cv2.resizeWindow('imageL '+alert1,700, 425)
            cv2.imshow('imageL '+alert1, tempLL)
            cv2.namedWindow('imageR '+alert1)
            cv2.resizeWindow('imageR '+alert1,700, 425)
            cv2.imshow('imageR '+alert1, tempRR)
            cv2.setMouseCallback('imageL '+alert1, draw_circle)
            if cv2.waitKey(20) & 0xFF ==ord('q'):
                break
        cv2.destroyAllWindows()
        
    def LF(self):
        imgLL.clear()
        filename,_=QFileDialog().getOpenFileName(self,"Load Image_L")
        if(filename==''): return
        temp= cv2.imread(filename)
        imgLL.append(temp)
    def LR(self):
        imgRL.clear()
        filename,_=QFileDialog().getOpenFileName(self,"Load Image_R")
        if(filename==''): return
        temp= cv2.imread(filename)
        imgRL.append(temp)
    def SWB(self):
        if self.lineEdit.text() != '':
            InputW=self.lineEdit.text()
        i=-1
        for img in images:
            temp=img.copy()
            i=i+1
            gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            cacu()
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            for x in range(len(InputW)) :
                fs = cv2.FileStorage('alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)
                ch = fs.getNode(InputW[x]).mat()
                if ((x+1)%3)==0 : 
                    movex=1
                else:
                    movex=7-(((x+1)%3)-1)*3
                if ((x+1)<4): 
                    movey=5
                else:
                    movey=2
                for l in range(len(ch)):
                    ch[l][0][0]=ch[l][0][0]+movex
                    ch[l][0][1]=ch[l][0][1]+movey
                    ch[l][1][0]=ch[l][1][0]+movex
                    ch[l][1][1]=ch[l][1][1]+movey
                    axis = np.float32(ch[l]).reshape(-1, 3)
                    imgpoints2, _ =cv2.projectPoints(axis,rvecs[i],tvecs[i],mtx,dist)
                    temp = draw1(temp,imgpoints2)
            cv2.namedWindow('Show word '+InputW+' on '+str(filenames[i]),0)
            cv2.resizeWindow('Show word '+InputW+' on '+str(filenames[i]),1080,900)
            cv2.imshow('Show word '+InputW+' on '+str(filenames[i]),temp)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

    def SWV(self):
        if self.lineEdit.text() != '':
            InputW=self.lineEdit.text()
        i=-1
        for img in images:
            i=i+1
            temp=img.copy()
            gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            cacu()
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            for x in range(len(InputW)) :
                fs = cv2.FileStorage('alphabet_lib_vertical.txt', cv2.FILE_STORAGE_READ)
                ch = fs.getNode(InputW[x]).mat()
                if ((x+1)%3)==0 : 
                    movex=1
                else:
                    movex=7-(((x+1)%3)-1)*3
                if ((x+1)<4): 
                    movey=5
                else:
                    movey=2
                for l in range(len(ch)):
                    ch[l][0][0]=ch[l][0][0]+movex
                    ch[l][0][1]=ch[l][0][1]+movey
                    ch[l][1][0]=ch[l][1][0]+movex
                    ch[l][1][1]=ch[l][1][1]+movey
                    axis = np.float32(ch[l])
                    imgpoints2, _ =cv2.projectPoints(axis,rvecs[i],tvecs[i],mtx,dist)
                    temp = draw1(temp,imgpoints2)
            cv2.namedWindow('Show word '+InputW+' on '+str(filenames[i]),0)
            cv2.resizeWindow('Show word '+InputW+' on '+str(filenames[i]),1080,900)
            cv2.imshow('Show word '+InputW+' on '+str(filenames[i]),temp)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()       
            
    def Undis(self):
        j=-1
        for img in images:
            j=j+1
            if(j==15): break
            gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            size = gray.shape
            w=size[1]
            h=size[0]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
            dst=cv2.cvtColor(dst,cv2.COLOR_RGB2GRAY)
            imgs=np.hstack((gray,dst))
            cv2.namedWindow('undistor '+str(filenames[j]),0)
            cv2.resizeWindow('undistor '+str(filenames[j]),1400,900)
            cv2.imshow('undistor '+str(filenames[j]),imgs)
            cv2.waitKey(750)
            cv2.destroyAllWindows()

    def Find_DI(self):
        img=images[0]
        gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print("1.4 Find the Distortion Matrix :")
        print(dist)
        print(" ")


    def Find_EI(self):
        value=int(self.comboBox.currentText())
        img=images[0]
        gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        rvecs,_=cv2.Rodrigues(rvecs[value-1])
        new_arr=np.concatenate([rvecs,tvecs[value-1]],1)
        print("1.3 Find the Extrinsic Matrix with picuture of " +str(value)+": ")
        print(new_arr)
        print(" ")

    def Find_IN(self):
        img=images[0]
        gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print("1.2 Find the Intrinsic Matrix : ")
        print(mtx)
        print(" ")
    def Find_corner(self):
        objpoints.clear()
        imgpoints.clear()
        i=-1
        for img in images:
            i=i+1
            gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            ret, corner=cv2.findChessboardCorners(gray, (11,8),None)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corner, (11,11), (-1,-1), criteria)
                imgpoints.append(corner)
                cv2.namedWindow('corner '+str(filenames[i]),0)
                cv2.resizeWindow('corner '+str(filenames[i]),700,900)
                temp=img.copy()
                cv2.drawChessboardCorners(temp, (11, 8), corners2, ret)
                cv2.imshow('corner '+str(filenames[i]),temp)
                cv2.waitKey(500)
                cv2.destroyAllWindows()

    def Load_Image(self):
        filename=QFileDialog().getExistingDirectory(self,"Load Image")
        if(filename==''): return
        load_images_from_folder(filename)

         


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = myMainWindow()
    window.show()
    sys.exit(app.exec_())