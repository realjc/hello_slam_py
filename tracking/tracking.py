import numpy as np 
import cv2

kAbsoluteScaleThreshold = 0.1        # absolute translation scale; it is also the minimum translation norm for an accepted motion 
kRansacThresholdNormalized = 0.0003  # metric threshold used for normalized image coordinates 
kRansacProb = 0.999

class VisualOdometry(object):
    def __init__(self, cam, grountruth, feature_tracker):
        self.cam = cam
        self.cur_image = None   # current image
        self.prev_image = None  # previous/reference image

        self.kps_ref = None  # reference keypoints 
        self.des_ref = None # refeference descriptors 
        self.kps_cur = None  # current keypoints 
        self.des_cur = None # current descriptors 

        self.cur_R = np.eye(3,3) # current rotation 
        self.cur_t = np.zeros((3,1)) # current translation 

        self.trueX, self.trueY, self.trueZ = None, None, None
        self.grountruth = grountruth     
        self.feature_tracker = feature_tracker

        self.init_history = True 
        self.traj3d_est = []         # history of estimated translations centered w.r.t. first one
        self.traj3d_gt = []          # history of estimated ground truth translations centered w.r.t. first one     


    def track(self, img, frame_id):
        print('..................................')
        print('frame: ', frame_id) 
        # convert image to gray if needed    
        if img.ndim>2:
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)             
        self.cur_image = img
        # manage and check stage 
        if frame_id>0:
            self.processFrame(frame_id)
        else:
            self.processFirstFrame()           
        self.prev_image = self.cur_image  

    def processFirstFrame(self):
        # only detect on the current image 
        self.kps_ref, self.des_ref = self.feature_tracker.detectAndCompute(self.cur_image)
        # convert from list of keypoints to an array of points 
        self.kps_ref = np.array([x.pt for x in self.kps_ref], dtype=np.float32) 


    def processFrame(self, frame_id):
        # track features 
        self.track_result = self.feature_tracker.track(self.prev_image, self.cur_image, self.kps_ref, self.des_ref)

        # estimate pose 
        R, t = self.estimatePose(self.track_result.kps_ref_matched, self.track_result.kps_cur_matched)     
        # update keypoints history  
        self.kps_ref = self.track_result.kps_ref
        self.kps_cur = self.track_result.kps_cur
        self.des_cur = self.track_result.des_cur 
 
        # t is estimated up to scale (i.e. the algorithm always returns ||trc||=1, we need a scale in order to recover a translation which is coherent with the previous estimated ones)
        absolute_scale = self.getAbsoluteScale(frame_id)
        if(absolute_scale > kAbsoluteScaleThreshold):
            # compose absolute motion [Rwa,twa] with estimated relative motion [Rab,s*tab] (s is the scale extracted from the ground truth)
            # [Rwb,twb] = [Rwa,twa]*[Rab,tab] = [Rwa*Rab|twa + Rwa*tab]
            self.cur_t = self.cur_t + absolute_scale*self.cur_R.dot(t) 
            self.cur_R = self.cur_R.dot(R)       

        # check if we have enough features to track otherwise detect new ones and start tracking from them (used for LK tracker) 
        if  (self.kps_ref.shape[0] < self.feature_tracker.num_features): 
            self.kps_cur, self.des_cur = self.feature_tracker.detectAndCompute(self.cur_image)           
            self.kps_cur = np.array([x.pt for x in self.kps_cur], dtype=np.float32) # convert from list of keypoints to an array of points       
            print('# new detected points: ', self.kps_cur.shape[0])                  
        self.kps_ref = self.kps_cur
        self.des_ref = self.des_cur
        self.updateHistory()     


    def estimatePose(self, kps_ref, kps_cur):	
        kp_ref_u = self.cam.undistort_points(kps_ref)	
        kp_cur_u = self.cam.undistort_points(kps_cur)	        
        self.kpn_ref = self.cam.unproject_points(kp_ref_u)
        self.kpn_cur = self.cam.unproject_points(kp_cur_u)

        # the essential matrix algorithm is more robust since it uses the five-point algorithm solver by D. Nister (see the notes and paper above )
        E, self.mask_match = cv2.findEssentialMat(self.kpn_cur, self.kpn_ref, focal=1, pp=(0., 0.), method=cv2.RANSAC, prob=kRansacProb, threshold=kRansacThresholdNormalized)

        #self.removeOutliersFromMask(self.mask)  # do not remove outliers, the last unmatched/outlier features can be matched and recognized as inliers in subsequent frames                          
        _, R, t, mask = cv2.recoverPose(E, self.kpn_cur, self.kpn_ref, focal=1, pp=(0., 0.))   
        return R,t  # Rrc, trc (with respect to 'ref' frame) 		

    # get current translation scale from ground-truth if grountruth is not None 
    def getAbsoluteScale(self, frame_id):  
        if self.grountruth is not None :
            self.trueX, self.trueY, self.trueZ, scale = self.grountruth.getPoseAndAbsoluteScale(frame_id)
            return scale
        else:
            self.trueX = 0 
            self.trueY = 0 
            self.trueZ = 0
            return 1

    def updateHistory(self):
        if (self.init_history is True) and (self.trueX is not None):
            self.t0_est = np.array([self.cur_t[0], self.cur_t[1], self.cur_t[2]])  # starting translation 
            self.t0_gt  = np.array([self.trueX, self.trueY, self.trueZ])           # starting translation 
            self.init_history = False 
        if (self.t0_est is not None) and (self.t0_gt is not None):             
            p = [self.cur_t[0]-self.t0_est[0], self.cur_t[1]-self.t0_est[1], self.cur_t[2]-self.t0_est[2]]   # the estimated traj starts at 0
            self.traj3d_est.append(p)
            pg = [self.trueX-self.t0_gt[0], self.trueY-self.t0_gt[1], self.trueZ-self.t0_gt[2]]  # the groudtruth traj starts at 0  
            self.traj3d_gt.append(pg)    