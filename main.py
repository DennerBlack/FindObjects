import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import cv2
import numpy as np
from PIL import ImageOps, Image

image = cv2.imread('data/images/target/Screenshot_1.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (3, 3), 0)

thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 11)
cv2.imshow('th', thresh)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

circles = []
rectangles = []
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
    if len(approx) == 4:
        rectangles.append(approx)

cv2.drawContours(image, circles, -1, (0, 255, 0), 2)

cv2.imshow('Detected Circles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()



