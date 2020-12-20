import sys
import os
currentPath = os.getcwd()
sys.path.append(currentPath)

import numpy as np
import yaml
import threading

from camera.camera import Camera
from dataset.dataset import VideoDataset
from slam.slam import SLAM



if __name__ == "__main__":
    configFilePath = "config/kitti/kitti06.yaml"
    with open(configFilePath,'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    camSettingFile = config['camera_settings_file']
    datasetVideo = config['dataset_file']
    cam = Camera(camSettingFile)
    dataset = VideoDataset(datasetVideo)
    slam = SLAM(cam=cam,dataset = dataset)
    slam.initilization()
    if not slam.isInitialized:
        print("初始化失败...")
        exit()
    print("初始化成功...")

    trackingThread = threading.Thread(target=slam.tracking, args=())
    trackingThread.start()
    mappingThread = threading.Thread(target=slam.mapping, args=())
    mappingThread.start()

