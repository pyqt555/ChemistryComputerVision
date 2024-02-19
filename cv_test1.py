import cv2 as cv2

import numpy as np 
import time
import math
import matplotlib.pyplot as plt

bisze=31
bstd=5



def updateBstd(value):
    global bstd
    bstd=value
def updateBsize(value):
    global bsize
    bsize=value
cv2.namedWindow("frame")
cv2.createTrackbar('Blur Standarddeviation', "frame", 0, 100, updateBstd)
cv2.createTrackbar('Blur Kernel size', "frame", 0, 500, updateBsize)





def high_cont(mat):
    pre_max=np.max(mat)#ermitteln des Maximums
    mat=mat.sum(-1)-50# -50

    mat=mat/np.max(mat)
    mat=cv2.GaussianBlur(mat,(21,21),6)#Weichzeichnen
    return mat*pre_max #Maximum wiederherstellen




cap = cv2.VideoCapture("vid_cam_02.mp4")#my webcam(2),default(0)

_, last =cap.read()
last=last[:,1000:-1]
last =cv2.rotate(last, cv2.ROTATE_90_COUNTERCLOCKWISE)
all_whites=[]
try:
    while(1):
        # Take each frame
        _, frame = cap.read()
        frame=frame[:,1000:-1]
        frame=cv2.rotate(frame,cv2.ROTATE_90_COUNTERCLOCKWISE)
        #t_arr=(frame-last)**2**1/2
        #t_arr=high_cont(t_arr)
        #t_arr=cv2.cvtColor(t_arr,cv2.COLOR_BGR2GRAY)
        #print(t_arr.shape)
        b_arr=(cv2.GaussianBlur(frame,(bisze,bisze),bstd)-cv2.GaussianBlur(last,(bisze,bisze),bstd))**2**1/2
        b_arr=high_cont(b_arr)
        last=frame


        cv2.imshow('frame',frame)
        #cv2.imshow('d_arr',t_arr)
        cv2.imshow("b_diff_arr",b_arr)
        cv2.imshow("b_arr_col",cv2.GaussianBlur(frame,(bisze,bisze),bstd))
        whites=np.where(b_arr==1)

        all_whites.append(whites)
        #cv.imshow('mask',mask)
        #cv.imshow('res',res)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
        
    cv2.destroyAllWindows()
except:
    pass


plt.plot(all_whites[50])
plt.show()



