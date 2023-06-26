import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
target_imgs = 'data/images/target'
val_imgs = 'data/images/validation'
raw_path = 'data/dataset/test/raw/'
kernel_size = (512, 512)

def cut_image(image, window_size):
	dims = np.int64(np.ceil(np.asarray([image.shape[0] / window_size[0], image.shape[1] / window_size[0]])))
	# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	images = []
	for i in range(dims[0]):
		for j in range(dims[1]):
			img = np.zeros(window_size+(3,))
			crop_img = image[window_size[0] * i:window_size[0] * (i + 1), window_size[1] * j:window_size[1] * (j + 1), :]
			img[:crop_img.shape[0], :crop_img.shape[1], :] += crop_img
			images.append(img)
			cv2.imwrite(raw_path + file[:-4] + '_' + str(i) + str(j) + '.png', img)
	return images

for file in listdir(target_imgs):
	if isfile(join(target_imgs, file)) and 'png' in file:
		image = cv2.imread(target_imgs+"/"+file)
		cut_image(image, kernel_size)






