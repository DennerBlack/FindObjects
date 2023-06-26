import os
import cv2
import numpy as np
from SCNN import SCNN


image_path = r'data/images/validation/Screenshot_6.png'
weights = r'weights/scnn_E300_v0_hard.h5'

image = cv2.imread(image_path)
detector = SCNN(weights)
detector.predict(image)



