import math
import matplotlib.pyplot as plt
import cv2
import numpy as np

from SignLib import custom_sort, get_center, get_beg_point, get_polar_coordinates_list, polar_to_decart

img = cv2.imread("/home/den/work/CV/FindObjects/data/images/target/Screenshot_1.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = 100

#get threshold image
ret,thresh_img = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

# find contours without approx
contours,_ = cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
contours=list(contours)
contours.sort(key=custom_sort)
sel_countour=contours[1]
xc,yc=get_center(sel_countour)
beg_point=get_beg_point(sel_countour,xc,yc)
polar_coord=get_polar_coordinates_list(sel_countour,xc,yc,beg_point)
count=100
full_angle=2*math.pi
i=1
end_angle = float(i) * full_angle / float(count)
summ=0.0
count_angles=0.0
signature=[]
for item_coord in polar_coord:
    angle,r=item_coord
    if angle>end_angle:
        signature.append((angle,summ/count_angles))
        i+=1
        end_angle = float(i) * full_angle / float(count)
        summ=0
        count_angles=0
    summ+=r
    count_angles+=1
signature.append((angle,summ/count_angles))
print(signature)

img_contours = np.zeros((img.shape[0],img.shape[1],3), np.uint8) # np.uint8(np.zeros((img.shape[0],img.shape[1])))
cv2.drawContours(img_contours, [sel_countour], -1, (255,0,0), 1)

for i in range(1,len(signature)):
    angle1,r1=signature[i-1]
    angle2,r2=signature[i]
    x1, y1=polar_to_decart(angle1,r1)
    x2, y2 = polar_to_decart(angle2, r2)
    cv2.line(img_contours, (int(x1+xc), int(y1+yc)), (int(x2+xc), int(y2+yc)), (0,0,255), thickness=1)
angle1,r1=signature[len(signature)-1]
angle2,r2=signature[0]
x1, y1=polar_to_decart(angle1,r1)
x2, y2 = polar_to_decart(angle2, r2)
cv2.line(img_contours, (int(x1+xc), int(y1+yc)), (int(x2+xc), int(y2+yc)), (0,0,255), thickness=1)

cv2.imshow('origin', img) # выводим итоговое изображение в окно
cv2.imshow('res', img_contours) # выводим итоговое изображение в окно

x=[]
y=[]
for item in signature:
    angle,r=item
    x.append(angle)
    y.append(r)

plt.plot(x,y)
plt.show()

cv2.waitKey()
cv2.destroyAllWindows()