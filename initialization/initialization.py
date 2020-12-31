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
        F, _ = cv2.findFundamentalMat(self.refKps, self.currKps, cv2.FM_RANSAC)
        if(F is None)
            self.initialized = False
        self.initialized = True
        return

    def reconWithF(self):
        

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





