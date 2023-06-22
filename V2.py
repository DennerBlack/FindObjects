import cv2
import numpy as np
from PIL import ImageOps, Image

image = cv2.imread('/home/den/work/CV/FindObjects/data/images/target/Screenshot_1.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (3, 3), 0)

thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 11)

_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours of white regions
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter and collect potential crosshair points
crosshair_points = []
for contour in contours:
    # Filter contours based on area or number of pixels
    if cv2.contourArea(contour) > 20:
        # Calculate the centroid of the contour
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        crosshair_points.append((cx, cy))

# Draw the detected crosshair points on the original image
for point in crosshair_points:
    cv2.circle(image, point, 5, (0, 255, 0), -1)

cv2.imshow('Detected Intersections', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

