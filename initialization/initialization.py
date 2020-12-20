import cv2
import numpy as np
import threading


class Initialization:
    def __init__(self):
        self.currFrame = None
        self.refFrame = None
        self.currKps = None
        self.refKps = None
        self.cam = None
        self.initialized = False
        self.H = []
        self.F = []



    def initialize(self):
        self.computeSIFTMatches()
        fScore ,retF = self.calcFScore()
        hScore ,retH = self.calcHScore()
        if(not retF  or not retH):
            self.initialized = False
            return
        rh = hScore/(fScore+hScore)
        if rh>0.40:
            self.reconWithH()
        else:
            self.reconWithF()
        self.initialized = True
        return


    def computeSIFTMatches(self):
        MIN_MATCH_COUNT = 10
        sift = cv2.SIFT_create()
        kps1,des1 = sift.detectAndCompute(self.refFrame, None)
        kps2, des2 = sift.detectAndCompute(self.currFrame,None)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        good = []
        for m,n in self.matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        if len(good)<MIN_MATCH_COUNT:
            self.initialized = False
            return

        self.refKps = np.float32([ kps1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        self.currKps = np.float32([ kps2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        
    def calcHScore(self):
        self.H, _ = cv.findHomography(self.refKps, self.currKps, cv.RANSAC,5.0)
        score = .0
        for kp1, kp2 in zip(self.refKps,self.currKps):
            err = np.linalg.norm(kp2[0]-(self.H[0,0]*kp1[0]+self.H[0,1]*kp1[1]+self.H[0,2])/ \
                    (self.H[2,0]*kp1[0]+self.H[2,1]*kp1[1]+self.H[2,2]),kp2[1]-(self.H[1,0]*kp1[0]+ \
                        self.H[1,1]*kp1[1]+self.H[2,2])/(self.H[2,0]*kp1[0]+self.H[2,1]*kp1[1]+ \
                            self.H[2,2]))
            if err<5.991:
                score += err
        return score, True



    def calcFScore(self):
        F, _ = cv2.findFundamentalMat(self.refKps, self.currKps, cv2.FM_RANSAC)
        if F is None or F.shape == (1, 1):
            return 0,False
        elif F.shape[0] > 3:
            F = F[0:3, 0:3]
        self.F = F
        for kp1, kp2 in zip(self.refKps,self.currKps):
            err = np.linalg.norm(np.append(kp1,np.array([1]))@F@np.append(kp2,np.array([1])))
            if err<5.991:
                score += err
        return score, True
        
        


