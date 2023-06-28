import cv2
import numpy as np
from os import listdir
from os.path import isfile, join


# это простой инструмент для нарезки изображений, для подготовки своего набора обучающих данных
# нужно указать путь к изображениям и путь сохранения нарезанных кусков
# kernel_size - размер окна нарезки, именно такое разрешение будет у кажого обрезанного куска


target_imgs = 'data/images/target'
save_path = 'data/dataset/test/raw/'
kernel_size = (512, 512)


def cut_image(image, window_size, save=False ,save_path=None):
	dims = np.int64(np.ceil(np.asarray([image.shape[0] / window_size[0], image.shape[1] / window_size[0]])))
	images = []
	for i in range(dims[0]):
		for j in range(dims[1]):
			img = np.zeros(window_size+(3,))
			crop_img = image[window_size[0] * i:window_size[0] * (i + 1), window_size[1] * j:window_size[1] * (j + 1), :]
			img[:crop_img.shape[0], :crop_img.shape[1], :] += crop_img
			images.append(img)
			if save:
				cv2.imwrite(save_path + file[:-4] + '_' + str(i) + str(j) + '.png', img)
	return images


if __name__ == '__main__':
	for file in listdir(target_imgs):
		if isfile(join(target_imgs, file)) and 'png' in file:
			image = cv2.imread(target_imgs+"/"+file)
			cut_image(image, kernel_size, save=True, save_path=save_path)






