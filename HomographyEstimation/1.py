import numpy as np
import cv2 as cv

def get_sift_correspondences(img1, img2, kpairs, ratio):
   
    sift =cv.SIFT_create(sigma=1.8)        
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    ############################################3
    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    ##########################################
    good_matches = sorted(good_matches, key=lambda x: x.distance)

    bestK_matches=[]
    for i in range(kpairs):
        bestK_matches.append(good_matches[i])

    points1 = np.array([kp1[m.queryIdx].pt for m in bestK_matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in bestK_matches])


    
    img_draw_match = cv.drawMatches(img1, kp1, img2, kp2, bestK_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    scale_percent = 30 # percent of original size
    width = int(img_draw_match.shape[1] * scale_percent / 100)
    height = int(img_draw_match.shape[0] * scale_percent / 100)
    dim = (width, height)
  
    # resize image
    resized = cv.resize(img_draw_match, dim, interpolation = cv.INTER_AREA)

    cv.imshow('match', resized)
    cv.waitKey(0)

    
    return points1, points2

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


def calerror(homo,npyfile):
    #####################################
    truth = np.load(npyfile)
    #######################
    exp1A=truth[0]
    exp1B=truth[1]

    totalerror=0

    for i in range(len(exp1A)):
        pp=exp1A[i]
        pp=list(pp)
        pp2=exp1B[i]
        pp2=list(pp2)
        ###############
        pp.append(1)
        pp2.append(1)
        ###############
        predict=np.matmul(homo,np.array(pp).transpose())
        newp=[]
        newp.append((predict[0]/predict[2]))
        newp.append((predict[1]/predict[2]))

        error=np.sqrt(((newp[0]-pp2[0])**2+(newp[1]-pp2[1])**2))

        totalerror+=error

    totalerror=totalerror/len(exp1A)

    return totalerror


def normalize(pp):
    x1bar=0
    y1bar=0

    for i in range(len(pp)):

        x1bar+=(pp[i])[0]
        y1bar+=(pp[i])[1]

    x1bar=x1bar/len(pp)
    y1bar=y1bar/len(pp)

    newp1s=[]
    dist1=0

    for i in range(len(pp)):
        temp=[]
        
        x1dist=(pp[i])[0]-x1bar
        temp.append(x1dist)
        
        y1dist=(pp[i])[1]-y1bar
        temp.append(y1dist)
        
        newp1s.append(temp)
        dist1+=np.sqrt((x1dist)**2+(y1dist)**2)

    dist1=dist1/len(pp)

    newp1s=np.array(newp1s)*(np.sqrt(2)/dist1)

    return newp1s

############################################
def myfunction(im0,im1,groundtruth,KPAIRS,RATIO,tonormalize):
    img0 = cv.imread(im0)
    img1 = cv.imread(im1)
    p1, p2 = get_sift_correspondences(img0, img1, KPAIRS, RATIO)

    if tonormalize:
        np1=normalize(p1)
        np2=normalize(p2)

        nhomo=get_homography_matrix(np1,np2)
        T=get_homography_matrix(p1,np1)
        T2=get_homography_matrix(p2,np2)

        homo=np.matmul(np.linalg.inv(T2),np.matmul(nhomo,T))

    else:
        homo=get_homography_matrix(p1,p2)


    error=calerror(homo,groundtruth)

    return error


######################################
aa0='1-0.png'
aa1='1-1.png'
aa2='1-2.png'

groundtruth1='correspondence_01.npy'
groundtruth2='correspondence_02.npy'

ratios=0.71
kpairs=[4,8,20]
#################################
print('Without Normalized:')
tonormalize=False
print('MSE for image A to image B:')
for i in range(len(kpairs)):
    myk=kpairs[i]
    myerror=myfunction(aa0,aa1,groundtruth1,myk,ratios,tonormalize)
    print(f"k= {myk}, error= {myerror}")
    print('')


print('Without Normalized:')
tonormalize=False
print('MSE for image A to image C:')
for i in range(len(kpairs)):
    myk=kpairs[i]
    myerror=myfunction(aa0,aa2,groundtruth2,myk,ratios,tonormalize)
    print(f"k= {myk}, error= {myerror}")
    print('')
print('###################################')

print('Normalized:')
tonormalize=True
print('MSE for image A to image B:')
for i in range(len(kpairs)):
    myk=kpairs[i]
    myerror=myfunction(aa0,aa1,groundtruth1,myk,ratios,tonormalize)
    print(f"k= {myk}, error= {myerror}")
    print('')


print('Normalized:')
tonormalize=True
print('MSE for image A to image C:')
for i in range(len(kpairs)):
    myk=kpairs[i]
    myerror=myfunction(aa0,aa2,groundtruth2,myk,ratios,tonormalize)
    print(f"k= {myk}, error= {myerror}")
    print('')
