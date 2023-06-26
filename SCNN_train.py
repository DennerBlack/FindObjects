import os
from typing import Union
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from keras.utils import array_to_img
import numpy as np
import cv2
import keras
from keras import layers
from keras import backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from os import listdir
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def add_noise(img):
    mean = 0
    stddev = 3
    noise = np.zeros(img.shape, np.uint8)
    cv2.randn(noise, mean, stddev)

    return cv2.add(img, noise)


def get_dataset(dir):
    masks = []
    images = []
    files_labels = listdir(dir+'/labeled')
    raw_files = listdir(dir+'/raw')
    raw_files.sort()
    for i, file in enumerate(raw_files):
        if 'png' in file:
            image = cv2.imread(dir+'/raw'+'/'+file)
            blur_img = cv2.GaussianBlur(image, (3, 3), 0)
            noise = add_noise(image)
            blur_noise = add_noise(blur_img)
            '''cv2.imshow('im1', noise)
            cv2.waitKey(0)
            cv2.destroyAllWindows()'''
            images.extend([image, blur_img, noise, blur_noise])
            if f'{i+1}.png' in files_labels:
                masks.extend([cv2.imread(dir+'/labeled'+'/'+f'{i+1}.png', cv2.IMREAD_GRAYSCALE)/255 for k in range(4)])
            else:
                masks.extend([np.zeros(input_image_size) for k in range(4)])
    return images, masks


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x

    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])
        previous_block_activation = x

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])
        previous_block_activation = x

    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    model = keras.Model(inputs, outputs)
    return model


def image_resize(image: np.ndarray, width=None, height=None, inter=None) -> Union[np.array, None]:
    inter = cv2.INTER_AREA
    dim = None
    try:
        (h, w) = image.shape[:2]
    except AttributeError:
        return None
    if width is None and height is None:
        return image

    if width and height:
        return cv2.resize(image, (width, height), interpolation=inter)

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


def center_crop(im):
    center = im.shape
    min_dim = min(center[:2])
    x = center[1] / 2 - min_dim / 2
    y = center[0] / 2 - min_dim / 2
    return im[int(y):int(y+min_dim), int(x):int(x+min_dim)]


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


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Hyperparameters
keras.utils.set_random_seed(101)
n_epochs = 300
batch_size = 2
num_classes = 2     # one more than needed
input_image_size = (512, 512)
train_dif = 0       # 0 - normal | 1 - hard
if train_dif:
    train_dif = r'hard/'
else:
    train_dif = r'normal/'
dataDir = r'data/dataset/' + train_dif
load_model = 1

# Generate train and validation arrays
dataset = get_dataset(dataDir)
train_x, train_y = np.asarray(dataset[0]), np.asarray(dataset[1])
val_x, val_y = np.asarray([cv2.imread(dataDir+'val_source.png')]), \
                np.asarray([cv2.imread(dataDir+'val_predict.png', cv2.IMREAD_GRAYSCALE)/255])

model = None
img_index = 0
images_number = 0

if not load_model:
    img_index = 1
    images_number = 4
    # Build model
    model = get_model(input_image_size, num_classes)
    optimizer = "rmsprop"
    loss = "sparse_categorical_crossentropy"
    metrics = [
        'accuracy',
        dice_metric,
        jaccard_distance_loss,
    ]
    plt.subplot(1, images_number, 1)
    history = None
    i = 0

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    # Start training
    history = model.fit(train_x,
                        train_y,
                        validation_data=(val_x, val_y),
                        batch_size=batch_size,
                        epochs = n_epochs,
                        verbose = True)

    i = 0
    while os.path.exists(f"weights/scnn_E{n_epochs}_v{i}_{train_dif}.h5"):
        i += 1
    model.save(f'weights/scnn_E{n_epochs}_v{i}_{train_dif}.h5')

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{f"scnn_E{n_epochs}_v{i}_{train_dif}"} fitting history')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
else:
    images_number = 3
    i = 0
    while os.path.exists(f"weights/scnn_E{n_epochs}_v{i+1}_{train_dif}.h5"):
        i += 1
    print(f'load model: scnn_E{n_epochs}_v{i}_{train_dif}.h5')
    model = keras.models.load_model(f'weights/scnn_E{n_epochs}_v{i}_{train_dif}.h5',
                                    custom_objects={"dice_metric": dice_metric,
                                                    "jaccard_distance_loss": jaccard_distance_loss})


image_gen = cv2.imread(dataDir+'/'+'val_source.png')
image = image_gen
plt.subplot(1, images_number, img_index+1)
plt.imshow(image)
val_preds = model.predict(cv2.resize(image, input_image_size).reshape(1,*input_image_size,3))

plt.subplot(1, images_number, img_index+2)
cv2.imshow('im0', image)
cv2.imshow('im1', val_preds.reshape((512, 512, 2))[:,:,0]*255)
cv2.imshow('im2', val_preds.reshape((512, 512, 2))[:,:,1]*255)
cv2.waitKey(0)
cv2.destroyAllWindows()
mask_gen = np.uint8(val_preds.reshape((512, 512, 2))[:,:,0]*255)
plt.imshow(np.uint8(val_preds.reshape((512, 512, 2))[:,:,1]*255))
plt.subplot(1, images_number, img_index+3)

plt.imshow(mask_gen)
plt.show()



