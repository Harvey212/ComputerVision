from scipy.spatial.transform import Rotation as Rot
import pandas as pd
import numpy as np
import cv2
import math
import open3d as o3d
#########################################################################
class P3P:
    def __init__(self):
        pass
    
    def solveIT(self, immp, wp):
        
        #############################################################
        immp_ex = immp[:3]
        immp_ex = np.insert(immp_ex, 2, np.ones(len(immp_ex)), axis=1) 
        
        ###########################################################
        #compute distance required
        dis = np.zeros(3, dtype=float)
        X0, Y0, Z0, X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3 = wp.flatten()
        dis[0] = math.sqrt((X1 - X2) * (X1 - X2) + (Y1 - Y2) * (Y1 - Y2) + (Z1 - Z2) * (Z1 - Z2))
        dis[1] = math.sqrt((X0 - X2) * (X0 - X2) + (Y0 - Y2) * (Y0 - Y2) + (Z0 - Z2) * (Z0 - Z2))
        dis[2] = math.sqrt((X0 - X1) * (X0 - X1) + (Y0 - Y1) * (Y0 - Y1) + (Z0 - Z1) * (Z0 - Z1))
        #######################################################
        #compute cosine required       
        u0, v0, k0, u1, v1, k1, u2, v2, k2 = immp_ex.flatten()
        len0, len1, len2 = np.sqrt(np.power(immp_ex, 2).sum(axis=1))
        u3, v3 = immp[3]

        cos = np.zeros(3, dtype=float)
        cos[0] = (u1 * u2 + v1 * v2 + k1 * k2) / (len1 * len2)
        cos[1] = (u0 * u2 + v0 * v2 + k0 * k2) / (len0 * len2)
        cos[2] = (u0 * u1 + v0 * v1 + k0 * k1) / (len0 * len1)
        ################################################################
        #construct the equation from the slide and find possible solution
        lengths = self.solvelength(dis, cos)
        if len(lengths) == 0:
            return []
        ##############################################3
        #for each possible solution, use companion matrix method to find possible Rotation and translation.
        #Calculate the reprojection error and return the best rotation and translation given the input world and pixel coordinate
        reproj_errors = []
        Rs = []
        Ts = []
        for length in lengths:
            MM = np.tile(length,3).reshape(3,3).T * immp_ex
            R, T = self.align(MM, X0, Y0, Z0, X1, Y1, Z1, X2, Y2, Z2)
            
            ptt = np.dot(R, np.array([X3,Y3,Z3])) + T
            ptt = ptt / ptt[2]
            u3p, v3p = ptt[:2]
            
            reproj_error = (u3p - u3)**2 + (v3p - v3)**2
            reproj_errors.append(reproj_error)
            Rs.append(R)
            Ts.append(T)

        reproj_errors = np.array(reproj_errors)
        Rs = np.array(Rs)
        Ts = np.array(Ts)
        ##############################################################
        sorted_idx = np.argsort(reproj_errors)
        sorted_Rs = Rs[sorted_idx]
        sorted_Ts = Ts[sorted_idx]

        best_Rs = sorted_Rs[0]
        best_Ts = sorted_Ts[0]
        ###########################################
        
        rotation_c = Rot.from_matrix(best_Rs)
        rvec = rotation_c.as_rotvec()

        return [rvec, best_Ts]
    

    def jacob(self,A):
        Z = [0,0,0,0]
        U = np.eye(4, dtype=float).flatten()


        B = [A[0], A[5], A[10], A[15]]
        D = B.copy()
        

        for itr in range(50):
            summ = abs(A[1]) + abs(A[2]) + abs(A[3]) + abs(A[6]) + abs(A[7]) + abs(A[11])
            if (summ == 0):
                return D, U

            trr=(0.2*summ/16) if (itr<3) else 0

            for i in range(3):
                loc = 5*i + 1
                for j in range(i+1, 4):
                    Aij=A[loc]
                    eps=100*abs(Aij)
                    if (itr>3 and abs(D[i])+eps== abs(D[i]) and abs(D[j])+eps==abs(D[j])):
                        A[loc] = 0
                    elif (abs(Aij) > trr):
                        hh = D[j] - D[i]
                        if (abs(hh) + eps==abs(hh)):
                            t = Aij / hh
                        else:
                            theta=0.5*hh/Aij
                            t = 1 / (abs(theta) + math.sqrt(1+pow(theta,2)))
                            if (theta < 0):
                                t = -t

                        hh=t*Aij
                        Z[i]-=hh
                        Z[j]+=hh
                        D[i]-=hh
                        D[j]+=hh
                        A[loc]=0

                        c=1.0/math.sqrt(1+pow(t,2))
                        s=t*c
                        tau=s/(1.0+c)

                        for k in range(i):
                            g = A[k*4+i]
                            h = A[k*4+j]
                            A[k*4+i]=g-s*(h+g*tau)
                            A[k*4+j]=h+s*(g-h*tau)

                        for k in range(i+1,j):
                            g=A[i*4+k]
                            h=A[k*4+j]
                            A[i*4+k]=g-s*(h+g*tau)
                            A[k*4+j]=h+s*(g-h*tau)

                        for k in range(j+1,4):
                            g=A[i*4+k]
                            h=A[j*4+k]
                            A[i*4+k]=g-s*(h+g*tau)
                            A[j*4+k]=h+s*(g-h*tau)

                        for k in range(4):
                            g=U[k*4+i]
                            h=U[k*4+j]
                            U[k*4+i]=g-s*(h+g*tau)
                            U[k*4+j]=h+s*(g-h*tau)

                    loc += 1
            for i in range(4):
                B[i] += Z[i]
            D = B.copy()
            Z = [0,0,0,0]
        return D, U

    def align(self, MM, X0, Y0, Z0, X1, Y1, Z1, X2, Y2, Z2):
       
        R = np.zeros((3,3), dtype=float)
        T = np.zeros(3, dtype=float)
        ######################################################
        #centroids
        C_start = np.zeros(3, dtype=float)
        C_start[0]=(X0+X1+X2)/3
        C_start[1]=(Y0+Y1+Y2)/3
        C_start[2]=(Z0+Z1+Z2)/3
        C_end = MM.mean(axis=0)
        ###########################################################
        #covariance matrix s
        s = np.zeros(9, dtype=float)
        for j in range(3):
            s[0*3+j]=(X0*MM[0][j]+X1*MM[1][j]+X2*MM[2][j])/3-C_end[j]*C_start[0]
            s[1*3+j]=(Y0*MM[0][j]+Y1*MM[1][j]+Y2*MM[2][j])/3-C_end[j]*C_start[1]
            s[2*3+j]=(Z0*MM[0][j]+Z1*MM[1][j]+Z2*MM[2][j])/3-C_end[j]*C_start[2]

        Qs = np.zeros(16, dtype=float)
        Qs[0*4+0]=s[0*3+0]+s[1*3+1]+s[2*3+2]
        Qs[1*4+1]=s[0*3+0]-s[1*3+1]-s[2*3+2]
        Qs[2*4+2]=s[1*3+1]-s[2*3+2]-s[0*3+0]
        Qs[3*4+3]=s[2*3+2]-s[0*3+0]-s[1*3+1]

        Qs[1*4+0]=Qs[0*4+1]=s[1*3+2]-s[2*3+1]
        Qs[2*4+0]=Qs[0*4+2]=s[2*3+0]-s[0*3+2]
        Qs[3*4+0]=Qs[0*4+3]=s[0*3+1]-s[1*3+0]
        Qs[2*4+1]=Qs[1*4+2]=s[1*3+0]+s[0*3+1]
        Qs[3*4+1]=Qs[1*4+3]=s[2*3+0]+s[0*3+2]
        Qs[3*4+2]=Qs[2*4+3]=s[2*3+1]+s[1*3+2]

        evs, U = self.jacob(Qs)
        ####################################################################
        #find largest eigen value
        ev_max = max(evs)
        ind = evs.index(ev_max)
        ####################################################
        #quaternion
        q = np.array(U).reshape(4,4)[:,ind]

        #change from a quaternion to an orthogonal matrix
        q_square = np.power(q,2)
        q0_1 = q[0] * q[1]
        q0_2 = q[0] * q[2]
        q0_3 = q[0] * q[3]
        q1_2 = q[1] * q[2]
        q1_3 = q[1] * q[3]
        q2_3 = q[2] * q[3]

        R[0][0] = q_square[0] + q_square[1] - q_square[2] - q_square[3]
        R[0][1] = 2. * (q1_2 - q0_3)
        R[0][2] = 2. * (q1_3 + q0_2)

        R[1][0] = 2. * (q1_2 + q0_3)
        R[1][1] = q_square[0] + q_square[2] - q_square[1] - q_square[3]
        R[1][2] = 2. * (q2_3 - q0_1)

        R[2][0] = 2. * (q1_3 - q0_2)
        R[2][1] = 2. * (q2_3 + q0_1)
        R[2][2] = q_square[0] + q_square[3] - q_square[1] - q_square[2]

        ##########################################################
        for i in range(3):
            T[i] = C_end[i] - (R[i][0] * C_start[0] + R[i][1] * C_start[1] + R[i][2] * C_start[2])

        return R, T

    def solvepoly(self, para):
        res = []
        para = np.array(para, dtype=float)
        try:
            p = np.poly1d(para)
            r = np.roots(p)
            r = r[np.isreal(r)]
        except:
            return res


        for root in r:
            res.append(np.real(root))
        return res

    def solvelength(self, distances, cosines):
       
        #########################################################
        p, q, r = cosines * 2
        d_p = np.power(distances, 2)
        a, b = (d_p/d_p[2])[:2]

        a2 = a**2
        b2 = b**2
        p2 = p**2
        q2 = q**2
        r2 = r**2
        pr = p*r
        pqr = pr*q
        ########################################################
        #the four points should not be coplanar
        if (p2 + q2 + r2 - pqr - 1 == 0):
            return []

        ab = a * b
        a_2 = 2*a
        A = -2 * b + b2 + a2 + 1 + ab*(2 - r2) - a_2
        ###################################################################
        if (A == 0):
            return []

        a_4 = 4*a
        B = q*(-2*(ab + a2 + 1 - b) + r2*ab + a_4) + pr*(b - b2 + ab)
        C = q2 + b2*(r2 + p2 - 2) - b*(p2 + pqr) - ab*(r2 + pqr) + (a2 - a_2)*(2 + q2) + 2
        D = pr*(ab-b2+b) + q*((p2-2)*b + 2 * (ab - a2) + a_4 - 2)
        E = 1 + 2*(b - a - ab) + b2 - b*p2 + a2
        
        tem=(p2*(a-1+b) + r2*(a-1-b) + pqr - a*pqr)
        b0=b*tem*tem
        #################################################################################
        if (b0 == 0):
            return []
        real_roots = self.solvepoly([A, B, C, D, E])
        if len(real_roots) == 0:
            return []

        r3 = r2*r
        pr2 = p*r2
        r3q = r3 * q
        inv_b0 =1/b0

        lengths = []
        ###############################################################
        for x in real_roots:
            if (x <= 0):
                continue

            x2 = x**2
            b1 = ((1-a-b)*x2 + (q*a-q)*x + 1 - a + b) * (((r3*(a2 + ab*(2 - r2) - a_2 + b2 - 2*b + 1)) * x + (r3q*(2*(b-a2) + a_4 + ab*(r2 - 2) - 2) + pr2*(1 + a2 + 2*(ab-a-b) + r2*(b - b2) + b2))) * x2 + (r3*(q2*(1-2*a+a2) + r2*(b2-ab) - a_4 + 2*(a2 - b2) + 2) + r*p2*(b2 + 2*(ab - b - a) + 1 + a2) + pr2*q*(a_4 + 2*(b - ab - a2) - 2 - r2*b)) * x + 2*r3q*(a_2 - b - a2 + ab - 1) + pr2*(q2 - a_4 + 2*(a2 - b2) + r2*b + q2*(a2 - a_2) + 2) + p2*(p*(2*(ab - a - b) + a2 + b2 + 1) + 2*q*r*(b + a_2 - a2 - ab - 1)))

            if (b1 <= 0):
                continue

            y = inv_b0 * b1
            v = x2 + y*y - x*y*r

            if (v <= 0):
                continue

            Z = distances[2] / math.sqrt(v)
            X = x * Z
            Y = y * Z

            lengths.append([X,Y,Z])
        #####################################################################

        return lengths
####################################################################
######################################################
def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    ##find k means of the descriptors
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    ##so far we calculate the average of the descriptor vector for each point id
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    ## you combine the pointid + average descriptors with xyz rgb
    return desc

########################################
def L2norm(vec1,vec2):
    v1=vec1[0]
    v2=vec2[0]

    dis=0
    for i in range(3):
        dis+=(v1[i]-v2[i])**2

    return math.sqrt(dis)
####################################################
def undistort(uv_orig,distCoeffs,camera):

    uv_raw = np.zeros((len(uv_orig),1,2), dtype=np.float32)
    for i, kp in enumerate(uv_orig):
        uv_raw[i][0] = (kp[0], kp[1])
    
    uv_new =cv2.undistortPoints(src=uv_raw,cameraMatrix=camera,distCoeffs=distCoeffs)
        
    result = []
    for i, uv in enumerate(uv_new):
        result.append(list(uv_new[i][0]))

    

    return np.array(result)
#####################################################
def myReproject(points3D,R,T,camera,distort):

    k1=distort[0]
    k2=distort[1]
    p1=distort[2]
    p2=distort[3]

    fx=camera[0,0]
    fy=camera[1,1]
    cx=camera[0,2]
    cy=camera[1,2]


    extrinsic = np.insert(R, 3, T, axis=1)
    #print(extrinsic)
    reprojected_points=[]

    points3D=points3D.reshape((-1,3))

    for i in range(points3D.shape[0]):
        mypt=[points3D[i,0],points3D[i,1],points3D[i,2],1]
        after1=np.matmul(extrinsic, np.array(mypt).reshape((4,1))).reshape((3,1))
        #after2=np.matmul(camera,after1)
        temp=[after1[0,0],after1[1,0],after1[2,0]]

        xp=temp[0]/temp[2]
        yp=temp[1]/temp[2]

        rsq=pow(xp,2)+pow(yp,2)
        xpp=xp*(1+k1*rsq+k2*pow(rsq,2))+2*p1*xp*yp+p2*(rsq+2*pow(xp,2))
        ypp=yp*(1+k1*rsq+k2*pow(rsq,2))+p1*(rsq+2*pow(yp,2))+2*p2*xp*yp

        u=fx*xpp+cx
        v=fy*ypp+cy

        ###############################################
        temp=[]
        temp.append(u)
        temp.append(v)
        
        #######################################################

        reprojected_points.append(temp)


    return np.array(reprojected_points)
##############################################################
def calError(points2D,reprojected_points,inlier_width=8.0):
    inliers=0
    myerror=0

    for row in range(points2D.shape[0]):
        p1x=points2D[row,0]
        p1y=points2D[row,1]

        p2x=reprojected_points[row,0]
        p2y=reprojected_points[row,1]

        error=math.sqrt((p1x-p2x)**2+(p1y-p2y)**2)
        if error<inlier_width:
            inliers+=1
        myerror+=error

    avg_error=myerror/len(points2D)

    return inliers,avg_error
###################################################

def myPnPRansac(points2D, points3D, cameraMatrix, distCoeffs, iteration = 100):
    max_inliers=0
    best_error=math.inf
    best_T = 0
    best_rvec = 0
    undistorted_points2D=undistort(points2D,distCoeffs,cameraMatrix)
    #######################################################
    #print(points2D)
    #ransac procedure
    for i in range(iteration):
        rnd = np.random.choice(undistorted_points2D.shape[0], 4, replace=False)

        ##########################################
        #solve p3p
        p3p=P3P()
        output=p3p.solveIT(undistorted_points2D[rnd], points3D[rnd])
        #print(output)
        if len(output)==0:
            continue
        else:
            rvec,T=output
        ###############################################
        #to see reproject     
        rotq = Rot.from_rotvec(rvec.reshape(1,3)).as_quat()
        R = Rot.from_quat(rotq).as_matrix()
                       
        reprojected_points= myReproject(points3D, R[0], T,cameraMatrix,distCoeffs)
        #print(reprojected_points)
        inliers,avg_error=calError(points2D,reprojected_points)

        if inliers> max_inliers:
            max_inliers = inliers
            best_T = T
            best_rvec = rvec
            best_error=avg_error
    ##############################################################   
    #print(best_error)
    return best_rvec, best_T
####################################################################
def pnpsolver(query,model,cameraMatrix,distortion):
    ##xy, decriptor
    kp_query, desc_query = query
    ##xyz, descriptor
    kp_model, desc_model = model


    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_query,desc_model,k=2)

    #find correspondence
    gmatches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            gmatches.append(m)

    points2D = np.empty((0,2))
    points3D = np.empty((0,3))

    for mat in gmatches:
        query_idx = mat.queryIdx
        model_idx = mat.trainIdx
        points2D = np.vstack((points2D,kp_query[query_idx]))
        points3D = np.vstack((points3D,kp_model[model_idx]))


    #cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
    #distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])


    best_rvec, best_T=myPnPRansac(points2D, points3D, cameraMatrix,distortion)
    return best_rvec, best_T
##################################################################3
#for open3d point clouds
def load_point_cloud(points3D_df):

    xyz = np.vstack(points3D_df['XYZ'])
    rgb = np.vstack(points3D_df['RGB'])/255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    return pcd
def load_pyramid(mat):
    mg=0.3
    hh=0.5
    center=[0, 0, 0,1]
    c1=[hh,mg,mg,1]
    c2=[hh,-mg,mg,1]
    c3=[hh,-mg,-mg,1]
    c4=[hh,mg,-mg,1]

    center_=list(np.matmul(mat,np.array(center).transpose()))
    c1_=list(np.matmul(mat,np.array(c1).transpose()))
    c2_=list(np.matmul(mat,np.array(c2).transpose()))
    c3_=list(np.matmul(mat,np.array(c3).transpose()))
    c4_=list(np.matmul(mat,np.array(c4).transpose()))

    points = [center_,c1_,c2_,c3_,c4_] 
    lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points) 
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([1, 0, 0])

    return line_set


def get_transform_mat(rotation, translation, scale):
    r_mat = Rot.from_euler('xyz', rotation, degrees=True).as_matrix()
    scale_mat = np.eye(3) * scale
    transform_mat = np.concatenate([scale_mat @ r_mat, translation.reshape(3, 1)], axis=1)
    return transform_mat

#####################################################################
class myproject:
    def __init__(self):
        #############################################
        ##world to camera (p=K[R|T]X)
        #image id(no repeat)
        #name train valid
        #tx ty tz  (translation)
        #qw qx qy qz (rotation)
        self.images_df = pd.read_pickle("data/images.pkl")
        ###################################################
        ##whole pt including training and validating. NO XYZ,rgb
        #point id(has reapeat)
        #image id(the point id's corresponding image .1->many)
        #xy (xy coordinate of the pointid for each image)
        #descriptors (descriptors of the pointid for each image)
        self.point_desc_df = pd.read_pickle("data/point_desc.pkl")
        #####################################
        ##the whole pointid
        #pointid(no repeat, but not every id exists)
        #xyz(pointid and its xyz in the real world)
        #rgb(point id's rgb)
        self.points3D_df = pd.read_pickle("data/points3D.pkl")
        ####################################
        ##only part of the data as train data from point_desc.pkl
        #pointID (has repeat)
        #XYZ (same pointid will have same xyz in the world)
        #rgb (same pointid will have same rgb in the world)
        #imageid (1 pointid may correspond to many image id)
        #xy (the corresponding xy in the image)
        #descriptors (its corresponding descriptors)
        self.train_df = pd.read_pickle("data/train.pkl")
        ######################################################################
        # Process model descriptors
        ##pointid(no repeat), average descriptor for each id,xyz,rgb
        self.desc_df = average_desc(self.train_df, self.points3D_df)
        ##it contains all the xyz coordinate for the pt id set(no repeat)in train data
        self.kp_model = np.array(self.desc_df["XYZ"].to_list())
        ##contains only descriptors for each pointid
        self.desc_model = np.array(self.desc_df["DESCRIPTORS"].to_list()).astype(np.float32)

        ###############################
        self.cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])    
        self.distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])


        #################################################
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()

        
        #########################################################
        # load point cloud
        pcd = load_point_cloud(self.points3D_df)
        self.vis.add_geometry(pcd)
        

    def Query(self,idx):
        #######################################################
        # Load query keypoints and descriptors
        #here we can see the point ids we have for that query image, we see all its descriptors, xy attr
        points = self.point_desc_df.loc[self.point_desc_df["IMAGE_ID"]==idx]

        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)
        ######################################################
        # Find correspondance and solve pnp
        ##for the xy,descriptors we have for the point in query image, we can calculate its world coor. by the trained model
        rvec, tvec=pnpsolver((kp_query, desc_query),(self.kp_model, self.desc_model),self.cameraMatrix,self.distCoeffs)
        ground_truth = self.images_df.loc[self.images_df["IMAGE_ID"]==idx]
        ####################################################################
        #[[-0.63125361 -0.31659399  2.31187954]]
        tvec = tvec.reshape(1,3)
        tvec_gt = ground_truth[["TX","TY","TZ"]].values
        trans_error=L2norm(tvec,tvec_gt)
        #print(trans_error)
        ############################################
        ############################################################
        rotq = Rot.from_rotvec(rvec.reshape(1,3)).as_quat()
        rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values

        myR = Rot.from_quat(rotq).as_matrix()
        tarR=Rot.from_quat(rotq_gt).as_matrix()

        diff=np.matmul(tarR[0],np.linalg.inv(myR[0]))
        rr=Rot.from_matrix(diff).as_rotvec(degrees=True)
        degree_error=np.linalg.norm(rr)
        #print(degree_error)
        #########################################################
        transform_mat = np.concatenate([myR[0], tvec.reshape(3, 1)], axis=1)
       
        return trans_error,degree_error,transform_mat



    def calculate(self):

        temp_transErr=[]
        temp_degreeErr=[]

        start=5
        end=655

        for mm in range(start, end, 5):
            print(mm)
            myname="valid_img"+str(mm)+".jpg"
            myp=self.images_df.loc[self.images_df["NAME"]==myname]
            idd=(list(myp["IMAGE_ID"]))[0]
           
            trans_error,degree_error,transform_mat=self.Query(idd)
            temp_transErr.append(trans_error)
            temp_degreeErr.append(degree_error)
            self.vis.add_geometry(load_pyramid(transform_mat))
        

        mid_trans=np.median(temp_transErr)
        mid_angle=np.median(temp_degreeErr)

        print(f"median of translation differences from ground truth is {mid_trans}.")
        print(f"median of angle differences from ground truth is {mid_angle}.")


        self.showResult()

        

    def showResult(self):
        # just set a proper initial camera view
        vc = self.vis.get_view_control()
        vc_cam = vc.convert_to_pinhole_camera_parameters()
        initial_cam = get_transform_mat(np.array([7.227, -16.950, -14.868]), np.array([-0.351, 1.036, 5.132]), 1)
        initial_cam = np.concatenate([initial_cam, np.zeros([1, 4])], 0)
        initial_cam[-1, -1] = 1.
        setattr(vc_cam, 'extrinsic', initial_cam)
        vc.convert_from_pinhole_camera_parameters(vc_cam)

        self.vis.run()
        self.vis.destroy_window()
        

######################################################################
pro=myproject()
pro.calculate()
