#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np 
import cv2 
import transforms3d as t3d 
import os 
import math 
#import open3d as o3d 

class Frame:
    def __init__(self,R,t,image):
        self.R = R
        self.t = t
        self.pose = np.vstack((np.hstack((R,t.T)),np.array([[0,0,0,1]])))
        self.image = image
        self.K = None

class Param:
    def __init__(self,border = 20,initDepth = 3.0, initCov = 3.0, 
                        minCov= 0.1, maxCov=10,nccWindowSize = 3):
        self.border = border
        self.initDepth = initDepth
        self.initCov = initCov
        self.minCov = minCov
        self.maxCov = maxCov
        self.nccWindowSize = nccWindowSize


def getTwoFrames(refIndex,currIndex,path,file="/first_200_frames_traj_over_table_input_sequence.txt", ):
    with open(path+file, 'r') as f:
        lines = f.readlines()
        def getAFrame(index):
            data = lines[index].split() ##数据格式：图像文件名 tx, ty, tz, qx, qy, qz, qw 
            image = cv2.imread(path + "/images/" + data[0])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            pose = np.eye(4)
            rot = t3d.quaternions.quat2mat([float(data[7]), float(data[4]), float(data[5]), float(data[6])])
            trans = np.array([[float(data[1]), float(data[2]), float(data[3])]])
            return Frame(rot,trans,image)

        refFrame = getAFrame(refIndex)
        currFrame = getAFrame(currIndex)
    
    return refFrame,currFrame


class depthMapRecon:
    def __init__(self, refFrame,currFrame, param):
        self.param = param
        self.currFrame = currFrame
        self.refFrame = refFrame
        self.poseTCR = np.linalg.inv(self.currFrame.pose).dot(self.refFrame.pose)
        self.depth = np.full(self.refFrame.image.shape,self.param.initDepth)
        self.depthCov = np.full(self.refFrame.image.shape,self.param.initCov)
        

    def px2Cam(self,px,K):
        return np.array([(px[0]-K[0,2])/K[0,0], (px[1]-K[1,2])/K[1,1], 1])

    def cam2Px(self,pCam,K):
        return np.array([pCam[0]*K[0,0]/pCam[2]+K[0,2], pCam[1]*K[1,1]/pCam[2]+K[1,2]])

    def isInlier(self, px):
        border = self.param.border
        row, col = self.currFrame.image.shape[0], self.currFrame.image.shape[1]
        return px[0]>=border and px[0]+border<row and \
                px[1]>border and px[0]+ border< col

    # def readDatasetFiles(self, path,file="/first_200_frames_traj_over_table_input_sequence.txt" ):
    #     f = open(path+file) 
    #     lines = f.readlines()
    #     self.poses_ = np.empty([len(lines),4,4])
    #     for i, line in enumerate(lines):
    #         dataList = line.split()##数据格式：图像文件名 tx, ty, tz, qx, qy, qz, qw 
    #         self.color_image_files_ = np.append(self.color_image_files_, path +"/images/" + dataList[0])
    #         tx, ty, tz, qx, qy, qz, qw  = float(dataList[1]),float(dataList[2]), \
    #             float(dataList[3]),float(dataList[4]),float(dataList[5]), \
    #             float(dataList[6]),float(dataList[7])
    #         rot = t3d.quaternions.quat2mat([qw, qx, qy, qz])
    #         pose = np.eye(4)
    #         pose[:3,:3] = rot
    #         pose[:3,3] = np.array([tx,ty,tz]).T 
    #         self.poses_[i] = pose
    #     f.closed
    #     return 


    def updateDepth(self):
        curr = self.currFrame.image
        ref = self.refFrame.image
        border = self.param.border
        row, col = self.refFrame.image.shape[0], self.refFrame.image.shape[1]
        minCov, maxCov = self.param.minCov, self.param.maxCov
        for x in range(border, row-border):
            for y in range(border, col-border):
                if(self.depthCov[x,y]<=minCov or self.depthCov[x,y]>maxCov):
                    continue
                ptRef = np.array([x,y])
                ret, ptCurr = self.epipolarSearch(ptRef)
                if(not ret):
                    continue
                if(not self.calcuDepth(ptRef, ptCurr)):
                    continue
        return

    def epipolarSearch(self,ptRef):
        depthMu = self.depth[ptRef[0],ptRef[1]]
        depthCov = math.sqrt(self.depthCov[ptRef[0],ptRef[1]])
        fRef = self.px2Cam(ptRef,self.refFrame.K)
        fRef /= np.linalg.norm(fRef)
        poseTCR = self.poseTCR

        pxMeanCurr = self.cam2Px(poseTCR[:3,:3].dot(fRef*depthMu)+poseTCR[:3,3],self.currFrame.K)
        dMin = depthMu -3*depthCov
        dMax = depthMu +3*depthCov
        if(dMin<0.1):
            dMin = 0.1
        pxMinCurr = self.cam2Px(poseTCR[:3,:3].dot(fRef*dMin)+poseTCR[:3,3],self.currFrame.K)
        pxMaxCurr = self.cam2Px(poseTCR[:3,:3].dot(fRef*dMax)+poseTCR[:3,3], self.currFrame.K)

        epipolarLine = pxMaxCurr-pxMinCurr
        epipolarDirection = epipolarLine/np.linalg.norm(epipolarLine)
        halfLength = 0.5*np.linalg.norm(epipolarLine)
        if(halfLength>100):
            halfLength = 100
        bestNCC = -1.0
        bestPxCurr = np.array([0,0])
        pxCurr = np.array([0,0])
        for l in np.arange(-halfLength,halfLength,0.7):
            pxCurr = pxMeanCurr + l*epipolarDirection
            if(not self.isInlier(pxCurr)):
                continue
            ncc = self.computeNCCScore( ptRef, pxCurr)
            if(ncc>bestNCC):
                bestNCC = ncc
                bestPxCurr = pxCurr
        if(bestNCC<0.85):
            return False, bestPxCurr
        return True, bestPxCurr

    def computeNCCScore(self,ptRef, ptCurr):
        curr = self.currFrame.image
        ref = self.refFrame.image
        nccWindowSize = self.param.nccWindowSize
        sum_up  = sum_down  = sum_down_1 = sum_down_2 = 0.0
        for r in range(-nccWindowSize, nccWindowSize+1):
            for c in range(-nccWindowSize, nccWindowSize+1):
                ref_val_uv = ref[ptRef[0]+r, ptRef[1]+c]/255
                cur_val_uv = curr[int(ptCurr[0])+r, int(ptCurr[1])+c]/255
                sum_up += ref_val_uv* cur_val_uv
                sum_down_1 += ref_val_uv * ref_val_uv
                sum_down_2 += cur_val_uv * cur_val_uv
        sum_down = math.sqrt(sum_down_1 * sum_down_2)
        ncc = sum_up / sum_down
        return ncc

    def calcuDepth(self,ptRef, ptCurr):
        poseTRC = np.linalg.inv(self.poseTCR)
        fRef =  self.px2Cam(ptRef, self.refFrame.K)
        fRef = fRef/np.linalg.norm(fRef)
        fCurr = self.px2Cam(ptCurr,self.currFrame.K)
        fCurr = fCurr/np.linalg.norm(fCurr)

        t = poseTRC[:3,3]
        f2 = poseTRC[:3,:3].dot(fCurr)
        b = np.array([t.dot(fRef), t.dot(f2)])
        A = np.array([fRef.dot(fRef),-fRef.dot(f2),fRef.dot(f2),-f2.dot(f2)])
        d = A[0]*A[3] -A[1]*A[2]
        # if(math.fabs(d)<1e-5):
        #     return False
        lambdaVec = np.array([A[3]*b[0]-A[1]*b[1], -A[2]*b[0]+A[0]*b[1]])/d
        xm = lambdaVec[0] * fRef
        xn = lambdaVec[1]*f2 + t
        depthEstimation = np.linalg.norm((xm+xn)/2.0)
        if(depthEstimation>0.5):
            print(depthEstimation)
        self.guassFusion(ptRef,fRef, t, depthEstimation)
        return True

    def guassFusion(self,ptRef,fRef, t, depthEstimation):
        p = fRef*depthEstimation
        a = p-t
        tNorm = np.linalg.norm(t)
        aNorm = np.linalg.norm(a)
        alpha = np.arccos(fRef.dot(t)/tNorm)
        beta = np.arccos(-a.dot(t)/(aNorm*tNorm))
        betaPrime = beta + np.arctan(1/self.refFrame.K[0,0])
        gamma = np.pi -alpha -betaPrime
        pPrime = tNorm*np.sin(betaPrime)/np.sin(gamma)
        dCov = pPrime-depthEstimation
        dCov2 = dCov*dCov

        mu = self.depth[ptRef[0],ptRef[1]]
        sigma2 = self.depthCov[ptRef[0],ptRef[1]]
        muFuse = (dCov2*mu +sigma2*depthEstimation)/(sigma2+dCov2)
        sigmaFuse2 = (sigma2*dCov2)/(sigma2+dCov2)
        self.depth[ptRef[0],ptRef[1]] = muFuse
        self.depthCov[ptRef[0],ptRef[1]] = sigmaFuse2
        return 




# if __name__ == "__main__":
#     depObj = depthMapRecon(480, 640)
#     depObj.readDatasetFiles("data/test_data")
#     refIndex = 0
#     currIndex = 1
#     depObj.updateDepth(refIndex, currIndex)
#     depthImage = image = cv2.normalize(depObj.depth_, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
#     cv2.imshow("result",depthImage)
#     cv2.imwrite("result.png", depthImage)
#     cv2.waitKey(5000)
    # poseTCR = np.array([[0.999999,  -0.00137732, -0.000123937,7.69453e-05],
    #                     [0.00137722,     0.999999, -0.000782858,-0.000502043],
    #                     [0.000125015,  0.000782687   ,  1,-0.000238606],
    #                     [0,0,0,1]])
    # ptRef = np.array([20,264])
    # ptCurr = np.array([20.0114, 264.25])
    # depObj.depthFilter(ptCurr,ptCurr,poseTCR)
# pt_ref  20
# 264
# pt_curr 20.0144
#  264.25
# T_C_R    0.999999  -0.00137732 -0.000123937
#   0.00137722     0.999999 -0.000782858
#  0.000125015  0.000782687            1
#  7.69453e-05
# -0.000502043
# -0.000238606
# depth_estimation 0.538952

if __name__ == '__main__':
    path = "data/test_data"
    refIndex = 0
    currIndex = 1
    K = np.array([[481.2,0,319.5],[0,-480.0,239.5 ] ,[0,0,1]])
    refFrame, currFrame = getTwoFrames(refIndex,currIndex,path)
    refFrame.K = currFrame.K = K
    param = Param()
    depObj = depthMapRecon(refFrame, currFrame, param)
    depObj.updateDepth()
    depthImage = cv2.normalize(depObj.depth, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    cv2.imwrite("result.png", depthImage)
    # pcd = o3d.geometry.create_point_cloud_from_depth_image(depth, intrinsic = K)
    # # flip the orientation, so it looks upright, not upside-down
    # pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

    # draw_geometries([pcd]) 


