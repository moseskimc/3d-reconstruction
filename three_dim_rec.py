import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy import linalg

class Reconstruction3D:
    """3D Reconstruction

        We implement the class direct linear transformation (DLT)
        to reconstruct the 3D coordinates of matched points between
        a stereo pair of images using the sift algorithm

        We will later extend this class to scene reconstruction
        if time allows
        
        
        Attributes:
            K1, K2: intrinsic calib matrices for both cams
            d1, d2: distortion coefficients for both cams
            essR1, essR2, ess_t: rotation and translation mats for change of basis (cam1 -> cam2)
            imgPts1, imgPts2: list of matched keypoint tuples
            fund: fundamental matrix corresponding to image pair 
    """

    def __init__(self, K1, K2, d1, d2):
        """Constructor

            Args:
                K1, K2 (np.ndarray): intrinsic matrices for cam 1 and 2, resp.
                d1, d2 (np.array): distortion coefficients for cam 1 and 2, resp.
            
            Returns:
                None

        """

        # intrinsic params
        self.K1, self.K2 = K1, K2
        self.d1, self.d2 = d1, d2

        # rotation or change of basis matrices from cam1 to cam2
        self.EssR1, self.EssR2 = None, None
        # corresponding translation vector
        self.Ess_t = None
        # fundamental matrix
        self.fund = None
        # projection matrices
        self.proj1, self.proj2 = None, None
        
        # data field
        self.imgPts1, self.imgPts2 = None, None
        
        # triangulation result
        self.TriPts = None

    def compute_proj_matrices(self):
        """Compute projection matrices given image pair and calibration data
            Returns:
                tuple: tuple of np.ndarray type objects corresp. to proj matrices (cam1, cam2)
        """

        # set internal cam params
        # cam1_intsc, cam2_intsc = np.eye(3), np.eye(3)
        cam1_intsc, cam2_intsc = self.K1, self.K2

        # proj matrix for cam 1
        # since camera 1 serves as reference point
        # rotation and translation are
        R1, t1 = np.eye(3), np.zeros((3,1))
        # now stack to form ext matrix for cam 2
        cam1_extsc = np.hstack((R1, t1))
        # proj matrix for cam1
        proj_mat1 = cam1_intsc @ cam1_extsc

        # proj matrix for cam 2
        # compute essential matrix
        E = self.est_essential_matrix()
        self.EssR1, self.EssR2, self.Ess_t = cv2.decomposeEssentialMat(E)
        # reshape the translation vector
        t2 = self.Ess_t.reshape(3, 1)
        # use second rotation matrix
        R2 = self.EssR2
        # stack rotation vector with translation vector
        # and compute extrinsic mat for cam 2
        cam2_extsc = np.hstack((R2, t2))
        # now compute projection matrix for cam 2
        proj_mat2 = cam2_intsc @ cam2_extsc
        
        self.proj1, self.proj2 = proj_mat1, proj_mat2

    def est_essential_matrix(self):
        """Return essential matrix given stereo pair correspondences
            Returns:
                np.ndarray: essential matrix
        """
        return self.K1.T.dot(self.fund).dot(self.K2)


    def load_img_pair(self, path1, path2):
        """Load image pair to class object field
        """
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
        
        return img1, img2

    def process_img_pair_keypts(self, path1, path2, method, draw):
        """Generate image pair matching key points using SIFT

        Args:
            param path1 (str): path to first image in pair
            param path2 (str): path to second image in pair
            param method (str, optional): method used to detect matching points pairs. Defaults to 'sift'.
            param draw (bool, optional): whether to plot the matches in matplotlib. Defaults to False.
        """
        img1, img2 = self.load_img_pair(path1, path2)
        
        if method == 'sift':
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SIFT_create()
            kps1, desc1 = sift.detectAndCompute(gray1, None)
            kps2, desc2 = sift.detectAndCompute(gray2, None)
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)  

            flann = cv2.FlannBasedMatcher(index_params,search_params)
            matches = flann.knnMatch(desc1,desc2,k=2)


            matchesMask = [[0,0] for i in range(len(matches))]
            good = []
            imgPts1 = []
            imgPts2 = []

            for i,(m,n) in enumerate(matches):
                if m.distance < 0.6*n.distance:
                    matchesMask[i]=[1,0]
                    good.append(m)
                    imgPts2.append(kps2[m.trainIdx].pt)   # TODO: find out what m.trainIDx is
                    imgPts1.append(kps1[m.queryIdx].pt)
            print("matches: "+str(len(good)))
            
            imgPts1 = np.int32(imgPts1)
            imgPts2 = np.int32(imgPts2)

            
            # apply mask computed from the fund matrix
            self.fund, mask = cv2.findFundamentalMat(imgPts1, imgPts2, cv2.LMEDS)

            self.imgPts1 = imgPts1[mask.ravel()==1]
            self.imgPts2 = imgPts2[mask.ravel()==1]
            
        # draw correspondences
        # converting BGR to RGB for plotting
        if draw:
            disp1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            disp2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            draw_params = dict(matchColor = (0,255,0),
                singlePointColor = (255,0,0),
                matchesMask = matchesMask,
                flags = 0)

            img3 = cv2.drawMatchesKnn(disp1,kps1,disp2,kps2,matches,None,**draw_params)
            plt.figure(figsize = (20,20))
            plt.imshow(img3,),plt.show()
    
    def normalize_keypts(self):
        """Normalize matched keypoints
        """
        
        imgPts1 = self.imgPts1
        imgPts2 = self.imgPts2
        
        # first homogenize and reshape before normalization
        hom_img1= cv2.convertPointsToHomogeneous(imgPts1).reshape(-1,3)
        hom_img2= cv2.convertPointsToHomogeneous(imgPts2).reshape(-1,3)
        
        # normalize
        K1_inv, K2_inv = np.linalg.inv(self.K1), np.linalg.inv(self.K2)
        imgPts1_norm = K1_inv.dot(hom_img1.T).T
        imgPts2_norm = K2_inv.dot(hom_img2.T).T
        
        # nonhomegenize to get planar coordinates
        imgPts1_norm = cv2.convertPointsFromHomogeneous(imgPts1_norm).reshape(-1,2)
        imgPts2_norm = cv2.convertPointsFromHomogeneous(imgPts2_norm).reshape(-1,2)
        
        
        self.imgPts1 = imgPts1_norm
        self.imgPts2 = imgPts2_norm
        
    def triangulate(self, use_dlt=True):
        """Triangulate using cv2's built-in method
        """
        
        # compute projection matrices
        Mat1, Mat2 = self.proj1, self.proj2
        Pts1, Pts2 = self.imgPts1, self.imgPts2
        if use_dlt:
            tris = []
            # loop through points pairs
            for pt1, pt2 in zip(Pts1, Pts2):
                tri = self.compute_dlt_tri(pt1, pt2)
                tris.append(tri)
            self.TriPts = np.vstack(tris)
        else:
            imgPts_hom = cv2.triangulatePoints(Mat1, Mat2, self.imgPts1.T, self.imgPts2.T).T

            # non-homogenize
            self.TriPts = cv2.convertPointsFromHomogeneous(imgPts_hom).reshape(-1,3)   
    
    def compute_dlt_tri(self, pt1, pt2):
        """Return triangulated point for pair pt1, pt2 usingi DLT

        Args:
            pt1 (np.array): coordinate tuple on cam 1 image plane
            pt2 (np.array): coordinate tuple on cam 2 image plane

        Returns:
            np.array: triangulated triplet from pt1 and pt2
        """
        
        # read in proj matrices
        P1, P2 = self.proj1, self.proj2
        
        # homogeneous coefficient matrix
        A = [pt1[1]*P1[2,:] - P1[1,:],
             P1[0,:] - pt1[0]*P1[2,:],
             pt2[1]*P2[2,:] - P2[1,:],
             P2[0,:] - pt2[0]*P2[2,:]
            ]
        A = np.array(A).reshape((4,4))
        
        # find eigenvectors with eigenvalue close to zero
        # find svd decompostion of A.T(A)
        B = A.transpose() @ A
        U, s, Vh = linalg.svd(B, full_matrices=False)

        return Vh[3,0:3]/Vh[3,3]
        
    def compute_reproj_error(self):
        """Compute RMSE for projection onto image plane of cam 2"""
        
        # compute rotation vector corresponding to EssR2
        rot_vector, jacobian = cv2.Rodrigues(self.EssR2)
        # project onto second image plane
        proj_points, _ = cv2.projectPoints(self.TriPts, rot_vector, self.Ess_t, self.K2, None)
        # finally, compute and return error
        return np.mean(np.sqrt(np.sum((self.imgPts2-proj_points.reshape(-1,2))**2,axis=-1)))

    def plot_3d_points(self):
        """Plot 3D points using matplotlib"""        
        
        # unpack coordintates
        x, y, z = self.TriPts.T
        # initialize figure
        fig = plt.figure()
        # enable 3d projection setting
        ax = fig.add_subplot(projection='3d')

        # set plot type
        marker ='o'
        ax.scatter(x, y, z, marker=marker)

        # set axis labels
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        
        # display
        plt.show()
