import cv2
import numpy as np
import yaml

class Camera:
    def __init__(self,camSettingFile):
        with open(camSettingFile,'r') as stream:
            cam_settings = yaml.load(stream, Loader=yaml.FullLoader)
        self.width = cam_settings['Camera.width']
        self.height = cam_settings['Camera.height']
        self.fx = cam_settings['Camera.fx']
        self.fy = cam_settings['Camera.fy']
        self.cx = cam_settings['Camera.cx']
        self.cy = cam_settings['Camera.cy']
        self.Kinv = np.array([[1/self.fx,    0,-self.cx/self.fx],
                              [   0, 1/self.fy,-self.cy/self.fy],
                              [   0,    0,    1]])  
        self.D = np.array([cam_settings['Camera.k1'],
                                cam_settings['Camera.k2'],
                                cam_settings['Camera.p1'],
                                cam_settings['Camera.p2'],
                                0],dtype=np.float32)
        self.is_distorted = np.linalg.norm(self.D) > 1e-10


    def undistort_points(self, uvs):
        if self.is_distorted:
            uvs_undistorted = cv2.undistortPoints(np.expand_dims(uvs, axis=1), self.K, self.D, None, self.K)      
            return uvs_undistorted.ravel().reshape(uvs_undistorted.shape[0], 2)
        else:
            return uvs 

    def unproject_points(self, uvs):
        return np.dot(self.Kinv, add_ones(uvs).T).T[:, 0:2]

# turn [[x,y]] -> [[x,y,1]]
def add_ones(x):
    if len(x.shape) == 1:
        return add_ones_1D(x)
    else:
        return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

# turn [[x,y]] -> [[x,y,1]]
def add_ones_1D(x):
    #return np.concatenate([x,np.array([1.0])], axis=0)
    return np.array([x[0], x[1], 1])
    #return np.append(x, 1)