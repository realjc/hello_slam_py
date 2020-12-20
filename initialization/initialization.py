import cv2
import numpy as np
import threading


class Initialization:
    def __init__(self):
        self.currFrame = None
        self.refFrame = None
        self.cam = None


    def initialize(self):
        