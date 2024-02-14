import numpy as np
import cv2
from PIL import Image


img=cv2.imread('test.jpg',0)
kernal=np.array([(3,-1,-2),(0,-2,-3),(-1,-2,4)])

spanW=img.shape[1]-kernal.shape[1]+1
spanH=img.shape[0]-kernal.shape[0]+1

newimg=np.zeros((spanH,spanW))

for i in range(1,(img.shape[0]-1)):
	for j in range(1,img.shape[1]-1):
		MM=img[i][j]*kernal[1][1]
		ML=img[i][j-1]*kernal[1][0]
		MR=img[i][j+1]*kernal[1][2]
		UM=img[i-1][j]*kernal[0][1]
		UL=img[i-1][j-1]*kernal[0][0]
		UR=img[i-1][j+1]*kernal[0][2]
		DM=img[i+1][j]*kernal[2][1]
		DL=img[i+1][j-1]*kernal[2][0]
		DR=img[i+1][j+1]*kernal[2][2]

		newimg[i-1][j-1]=(MM+ML+MR+UM+UL+UR+DM+DL+DR)/9

new_im = Image.fromarray(newimg.astype(np.uint8))
new_im.show()
cv2.imshow('image',img)
cv2.waitKey()
