import numpy as np
import matplotlib.pyplot as plt
import cv2

Img = cv2.imread('jirafa.jpg')
Img_gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
neighborhood = 2
apperture = 3
alpha = 0.04

# Taking a matrix of size 5 as the kernel 
kernel = np.ones((5,5), np.uint8) 

dst = cv2.cornerHarris(Img_gray, neighborhood,apperture, alpha)
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,kernel, iterations=1) 
# Threshold for an optimal value, it may vary depending on the image.
Img[dst>0.01*dst.max()]=[0,255,0]
#print (dst.max())
cv2.imshow('dst',dst)
cv2.imshow('image',Img )
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()