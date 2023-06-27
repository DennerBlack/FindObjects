import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from SCNN import SCNN

# пути к изображениям
# первый - к изображению на котором нет маркеров
# второй - к "сложному" изображению
# третий - к "простому" изображению
# Сложность изображения - количество шумов в зоне маркера
image_path = r'data/images/validation/Screenshot_6.png'
image_path_hard = r'data/dataset/hard/val_source.png'
image_path_norm = r'data/dataset/normal/val_source.png'

# пути к весам моделей
# в данном детекторе я использую дифференциальный подход к генерации масок,
# то есть используются предсказания двух немного по-разному обученных моделей,
# после берется маска, полученная с помощью наложения двух исходных масок через логическое "И"
weights = r'weights/scnn_E300_v1_hard.h5'
diff_weights = r'weights/scnn_E300_v0_normal.h5'

# Параметр, указывающий чувствительность детектора - это минимальная ширина найденного объекта
# если меньше, то оно отфильтровывается
# из-за крайне малой выборки для обучения есть небольшой шум в генерируемых масках, его нужно фильтровать
threshold = 30

image = cv2.imread(image_path)
# инициализация детектора, указываются веса к моделям
detector = SCNN(weights, diff_weights)
# нахождение масок, в качестве параметра необходимо передать изображение, параметр чувствительности необязателен
mask, labeled_image = detector.predict(image, threshold=threshold)

plt.subplot(1, 2, 1)
plt.title('Исходное изображение')
plt.imshow(image)
plt.subplot(1, 2, 2)
plt.title('Сгенерированная детектором маска')
plt.imshow(mask)
plt.show()

plt.imshow(labeled_image)
plt.title('Размеченное изображение')
plt.show()

