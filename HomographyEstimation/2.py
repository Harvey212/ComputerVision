import numpy as np
import cv2 as cv


def get_homography_matrix(source, destination):
    
    A = []
    for i in range(len(source)):
        s_x, s_y = source[i]
        d_x, d_y = destination[i]
        #####################################
        A.append([0, 0, 0, -s_x, -s_y, -1, (d_y)*(s_x), (d_y)*(s_y), d_y])
        A.append([s_x, s_y, 1, 0, 0, 0, (-d_x)*(s_x), (-d_x)*(s_y), -d_x])   

    #w and vh don't have the same dimension
    U, w, vh=np.linalg.svd(A,full_matrices=True)

    #You just take the last one. You don't have to use vh[:,-1]. They do it for you
    ff=vh[-1]
    result=np.reshape(ff, (3, 3))
    result=result/result[2,2]
    
    return result
####################################################################    
def backwardWarp(c0,hoI,MAXH,MAXW):
    AA=np.zeros((MAXH,MAXW), dtype=np.uint8)#
    for i in range(MAXH):
        for j in range(MAXW):
            temp=[]
            temp.append(j)
            temp.append(i)
            temp.append(1)
            res = np.matmul(hoI, np.array(temp).transpose())

            gg=[]
            gg.append(res[0]/res[2])
            gg.append(res[1]/res[2])

            XX=gg[0]
            YY=gg[1]

            x1=int(XX)
            x2=int(XX)+1

            y1=int(YY)
            y2=int(YY)+1
            #################################
    
            a=c0[y1,x1]
            b=c0[y1,x2]
            c=c0[y2,x1]
            d=c0[y2,x2]

            wa=(x2-XX)*(y2-YY)
            wb=(XX-x1)*(y2-YY)
            wc=(x2-XX)*(YY-y1)
            wd=(XX-x1)*(YY-y1)

            ww=wa+wb+wc+wd
            ra=wa/ww
            rb=wb/ww
            rc=wc/ww
            rd=wd/ww

            rr=ra*a+rb*b+rc*c+rd*d
   
            AA[i,j]=rr

    return AA

###################################################################
img = cv.imread('withPoint.png')

d0=img[:,:,0]
d1=img[:,:,1]
d2=img[:,:,2]

##########################################################
#[col,row]
pp=[[36, 127], [191, 106], [113, 354], [272, 279]]

A=pp[0]
B=pp[1]
C=pp[2]
D=pp[3]

#############################
widthAB=np.sqrt((A[0]-B[0])**2+(A[1]-B[1])**2)
widthCD=np.sqrt((C[0]-D[0])**2+(C[1]-D[1])**2)

MAXW=max(int(widthAB),int(widthCD))

##############################################
heightAC=np.sqrt((A[0]-C[0])**2+(A[1]-C[1])**2)
heightBD=np.sqrt((B[0]-D[0])**2+(B[1]-D[1])**2)

MAXH=max(int(heightAC),int(heightBD))
############################################
after=[]
after.append([0,0])
after.append([MAXW-1,0])
after.append([0,MAXH-1])
after.append([MAXW-1,MAXH-1])
#############################
ddI=np.linalg.inv(get_homography_matrix(pp,after))
######################################
#you must add dtype=np.uint8
ss = np.zeros((MAXH,MAXW,3), dtype=np.uint8)

ss[:,:,0]=np.array(backwardWarp(d0,ddI,MAXH,MAXW))
ss[:,:,1]=np.array(backwardWarp(d1,ddI,MAXH,MAXW))
ss[:,:,2]=np.array(backwardWarp(d2,ddI,MAXH,MAXW))

cv.imshow("image",ss)
cv.waitKey(0)