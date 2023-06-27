import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import cv2
import keras
from keras import backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from Prepare import cut_image

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def dice_metric(y_true, y_pred):
    y_pred = y_pred[:,:,:,0]
    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
    union = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))

    return 2*intersection / union


def jaccard_distance_loss(x_true, x_pred, smooth=100):
    x_pred = x_pred[:,:,:,0]
    intersection = K.sum(K.sum(K.abs(x_true * x_pred), axis=-1))
    sum_ = K.sum(K.sum(K.abs(x_true) + K.abs(x_pred), axis=-1))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return (1 - jac) * smooth


def _clustering(image: np.ndarray, oi: np.ndarray, threshold: int):
    contours, hierarchy = cv2.findContours(cv2.GaussianBlur(np.uint8(image),(9,9),cv2.BORDER_DEFAULT),
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    num_of_objects = 0
    centers = []
    for cnt in contours:
        _, _, w, _ = cv2.boundingRect(cnt)
        if w < threshold: continue
        moments = cv2.moments(cnt)
        try:
            x = int(moments["m10"] / moments["m00"])
            y = int(moments["m01"] / moments["m00"])
        except:
            continue
        num_of_objects+=1
        centers.append((x,y))
        cv2.circle(np.uint8(oi), (x, y), 30, (255, 125, 0), 2)
        cv2.circle(np.uint8(oi), (x, y), 3, (0, 0, 255), -1)
        cv2.putText(np.uint8(oi), f'Маркер: {num_of_objects}', (x-30, y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return num_of_objects, centers


def _show_image_date(data):
    print(f'Кол-во найденных маркеров: {data[0]}')
    if data[0]:
        print(f'Центры найденных маркеров:')
        for center in data[1]:
            print(f'X={center[0]}, Y={center[1]}')


def _data_postprocessing(mask, image, threshold):
    objects_data = _clustering(mask, image, threshold)
    _show_image_date(objects_data)


class SCNN():
    def __init__(self, model_path: str, diff_model_path: str, verbose=False):
        self.model = keras.models.load_model(model_path,
                                             custom_objects={"dice_metric": dice_metric,
                                                             "jaccard_distance_loss": jaccard_distance_loss})
        self.diff_model = keras.models.load_model(diff_model_path,
                                                  custom_objects={"dice_metric": dice_metric,
                                                                  "jaccard_distance_loss": jaccard_distance_loss})
        self.verbose = verbose
        self.input_size = (512, 512)

    def predict(self, image: np.ndarray, threshold: int = 30):
        image = image.copy()
        if image.shape[:2] == self.input_size:
            mask_prediction = self._single_prediction(image)
        else:
            mask_prediction = self._multiple_predictions(image)
        _data_postprocessing(mask_prediction, image, threshold)

        return mask_prediction, image

    def _single_prediction(self, image: np.ndarray):
        mask_prediction = self.model.predict(cv2.resize(image, self.input_size).reshape(1, *self.input_size, 3),
                                             verbose=self.verbose)
        diff_mask = self.diff_model.predict(cv2.resize(image, self.input_size).reshape(1, *self.input_size, 3),
                                            verbose=self.verbose)
        mask_prediction = cv2.bitwise_and(mask_prediction, diff_mask)
        return mask_prediction.reshape((512, 512, 2))[:, :, 1] * 255

    def _multiple_predictions(self, image: np.ndarray):
        images = cut_image(image, self.input_size)
        source_size = image.shape[:2]
        dims = np.int64(np.ceil(np.asarray([image.shape[0] / self.input_size[0],
                                            image.shape[1] / self.input_size[0]])))

        canvas = np.zeros((self.input_size[0]*dims[0], self.input_size[1]*dims[1]))

        for i in range(dims[0]):
            for j in range(dims[1]):
                mask = self._single_prediction(images[i+dims[0]*j])
                canvas[i * self.input_size[0]:(i + 1) * self.input_size[0],
                      j * self.input_size[0]:(j + 1) * self.input_size[0]] = mask
        canvas = canvas[:source_size[0],:source_size[1]]

        return canvas

