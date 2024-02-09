import open3d as o3d
import numpy as np
import cv2 as cv
import sys, os, argparse, glob
import multiprocessing as mp

class SimpleVO:
    def __init__(self, args):
        camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
        self.K = camera_params['K']
        self.dist = camera_params['dist']
        self.frame_paths = sorted(list(glob.glob(os.path.join(args.input, '*.png'))))

        self.orb=cv.ORB_create()
        self.bf=cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        self.vis=o3d.visualization.Visualizer()
        self.vis.create_window(width=900,height=900)

        self.curr_pos=np.zeros((3, 1),dtype=np.float64)
        self.curr_rot=np.eye(3,dtype=np.float64)
    
    ################################################################
    def getLineset(self,center, coo):
        points = [center, coo[0],coo[1],coo[2],coo[3]]
        lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[1,4]]
        colors = [[1,0,0] for i in range(len(lines))]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    def get_pyramid(self,r_M,pos,cam_M):
        pos = pos.reshape(3)
        relativeCor = np.array([[0,0,1],[600,0,1],[600,600,1],[0,600,1]]).T
        realCor = np.dot(r_M, np.dot(np.linalg.inv(cam_M), relativeCor)).T + pos
        
        return pos,realCor
    ##############################################################
    def triangulation(self, R, t, pt1, pt2, K):
        ###################################
        pose1 = np.array([[1, 0, 0, 0], 
                          [0, 1, 0, 0], 
                          [0, 0, 1, 0]])
        pose1 = K.dot(pose1)
        ####################################
        pose2 = np.hstack((R, t))
        pose2 = K.dot(pose2)
        
        tripoints = cv.triangulatePoints(pose1,pose2,pt1.reshape(2,-1),pt2.reshape(2,-1)).reshape(-1, 4)[:,:3]
        
        return tripoints

    ###############################
    def getScale(self, old_3d, new_3d):


        temp=[]

        for i in range(len(old_3d)):
            shiftold3d=old_3d[(i+1)%len(old_3d)]
            oriold3d=old_3d[i]
            s1=(shiftold3d[0]-oriold3d[0])**2+(shiftold3d[1]-oriold3d[1])**2+(shiftold3d[2]-oriold3d[2])**2
            s1=np.sqrt(s1)


            shiftnew3d=new_3d[(i+1)%len(new_3d)]
            orinew3d=new_3d[i]

            s2=(shiftnew3d[0]-orinew3d[0])**2+(shiftnew3d[1]-orinew3d[1])**2+(shiftnew3d[2]-orinew3d[2])**2
            s2=np.sqrt(s2)
            temp.append(s1/s2)
        #############################
        ratio=np.median(temp)
               
        return ratio
    
    ###########################################################
    

    def process_frames(self):
        prev_img = cv.imread(self.frame_paths[0])
        for ind, frame_path in enumerate(self.frame_paths[1:]):
            #######################################################3
            #read image
            curr_img = cv.imread(frame_path)
            #curr_img = cv.undistort(curr_img, cameraMatrix=self.K, distCoeffs=self.dist)
            ####################################################
            #Feature Matching
            kp1,des1=self.orb.detectAndCompute(prev_img, None)
            kp2,des2=self.orb.detectAndCompute(curr_img, None)

            matches=self.bf.match(des2,des1)

            matches=sorted(matches,key=lambda x:x.distance)

            pts1 = np.float32([kp1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts2 = np.float32([kp2[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

            #estimate essential matrix Ek,k+1
            E,mask=cv.findEssentialMat(pts2, pts1, self.K, cv.RANSAC, 0.999, 1, None)
            _, R_curr,t_curr,mask=cv.recoverPose(E, pts2, pts1, cameraMatrix=self.K, mask=mask)

            if ind!=0:
                ################################################
                #store index to find common set of consecutive frame
                old_kp1_ind=[]

                old_kp2_ind=[]
                new_kp1_ind=[]

                new_kp2_ind=[]
            
                com_old_kp1_ind=[]
                #################################

                common= set([m.trainIdx for m in matches]).intersection([m.queryIdx for m in matches_prev])
            
                for m in matches_prev:
                    if (m.queryIdx in common):
                        old_kp1_ind.append(m.trainIdx)
                        old_kp2_ind.append(m.queryIdx)

                for m in matches:
                    if (m.trainIdx in common):
                        new_kp1_ind.append(m.trainIdx)
                        new_kp2_ind.append(m.queryIdx)
                        
                for m1 in new_kp1_ind:
                    m0 = old_kp2_ind.index(m1)
                    com_old_kp1_ind.append(old_kp1_ind[m0])


                pts0=np.float32([old_kp1[idx].pt for idx in com_old_kp1_ind]).reshape(-1, 1, 2)
                pts1=np.float32([kp1[idx].pt for idx in new_kp1_ind]).reshape(-1, 1, 2)
                pts2=np.float32([kp2[idx].pt for idx in new_kp2_ind]).reshape(-1, 1, 2)

                ######################################
                #find triangulation points 
                #goal: to find relative ratio
                old_tri = self.triangulation(R_prev, t_prev, pts0, pts1, self.K)
                new_tri = self.triangulation(R_curr, t_curr, pts1, pts2, self.K)
                

                ratio = self.getScale(old_tri, new_tri)
                #####################################################

            else:
                ratio=1
            

            #get the relative position to the first frame
            self.curr_pos+=self.curr_rot.dot(t_curr)*ratio
            self.curr_rot=R_curr.dot(self.curr_rot)
                    
            prev_img=curr_img
            matches_prev=matches
            old_kp1=kp1
            old_kp2=kp2
            R_prev=R_curr
            t_prev=t_curr
            
            #########################################################  
            #draw the matched points on the current image  
            img = cv.drawKeypoints(curr_img, kp2, None, color=(0,0,255))
            cv.imshow('frame', img)
            
            ############################
            #update the new camera pose in open3D window
            center, corners = self.get_pyramid(self.curr_rot,self.curr_pos, self.K)
            line_set = self.getLineset(center,corners)
            
            self.vis.add_geometry(line_set)
            self.vis.poll_events()

            if cv.waitKey(30) == 27: break
            
        #################################################
        cv.destroyWindow('frame')  
        self.vis.run()
        self.vis.destroy_window()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',default="./frames/",help='directory of sequential frames')
    parser.add_argument('--camera_parameters',default='camera_parameters.npy',help='npy file of camera parameters')
    args = parser.parse_args()

    vo = SimpleVO(args)
    vo.process_frames()