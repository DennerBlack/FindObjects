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


class SCNN():
    def __init__(self, model_path: str):
        self.model = keras.models.load_model(model_path,
                                    custom_objects={"dice_metric": dice_metric,
                                                    "jaccard_distance_loss": jaccard_distance_loss})
        self.input_size = (512, 512)

    def predict(self, image: np.ndarray) -> np.ndarray:
        if image.shape[:2] != self.input_size:
            mask_prediction = self._multiple_predictions(image)

        mask_prediction = self._single_prediction(image)
        return mask_prediction

    def _single_prediction(self, image: np.ndarray):
        mask_prediction = self.model.predict(cv2.resize(image, self.input_size).reshape(1, *self.input_size, 3))
        return mask_prediction.reshape((512, 512, 2))[:, :, 1] * 255

    def _multiple_predictions(self, image: np.ndarray):
        images = cut_image(image, self.input_size)
        source_size = image.shape[:2]
        dims = np.int64(np.ceil(np.asarray([image.shape[0] / self.input_size[0],
                                            image.shape[1] / self.input_size[0]])))

        canvas = np.zeros((self.input_size[0]*dims[0], self.input_size[1]*dims[1], 3))

        for i in dims[0]:
            for j in dims[1]:
                mask = self._single_prediction(images[i+dims[0]*j])
                canvas[i * self.input_size[0]:(i + 1) * self.input_size[0],
                      j * self.input_size[0]:(j + 1) * self.input_size[0], :] = mask
        cv2.imshow('im1', canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return canvas
