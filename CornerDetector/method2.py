import cv2
import numpy as np
import scipy.linalg as la

def findCorners(img, window_size, k, thresh,imm):
    #Find x and y derivatives
    dy, dx = np.gradient(img)

    mdx=np.mean(dx)
    sdx=np.std(dx)

    fdx=(dx-mdx)/sdx

    mdy=np.mean(dy)
    sdy=np.std(dy)

    fdy=(dy-mdy)/sdy

    Ixx = fdx**2
    Ixy = fdx*fdy
    Iyx = fdy*fdx
    Iyy = fdy**2

    height = img.shape[0]
    width = img.shape[1]

    newImg = imm.copy()
    
    offset = int(window_size/2)
    cons=1e-3
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIyx = Iyx[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]

            
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syx = windowIyx.sum()
            Syy = windowIyy.sum()


            cov=np.array([[Sxx,Sxy],[Syx,Syy]])

            eigvals, eigvecs = la.eig(cov)
            eigvals = eigvals.real
            lambda1 = eigvals[0]
            lambda2 = eigvals[1]

            det=lambda1*lambda2
            trace=lambda1+lambda2

            #r=det-k*(trace**2)
            r=det/(trace+cons)

            if r > thresh:
                newImg.itemset((y, x, 0), 0)
                newImg.itemset((y, x, 1), 0)
                newImg.itemset((y, x, 2), 255)
    return newImg


    

window_size = 3
k = 0.04
thresh = 10

img = cv2.imread('building.jpg')
im2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

gray = np.float32(im2)
finalImg= findCorners(gray, int(window_size), float(k), int(thresh),img)
        

cv2.imshow('dst',finalImg)

cv2.waitKey(0)
cv2.destroyAllWindows()



