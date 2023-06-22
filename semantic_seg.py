import os
import time
from typing import Union
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from keras.utils import array_to_img
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from os import listdir
from os.path import isfile, join
### For visualizing the outputs ###
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
matplotlib.use('TkAgg')
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

'''
dataDir = r'cubes_dataset/coco'

annFile = fr'{dataDir}/result.json'

coco = COCO(annFile)
catIDs = coco.getCatIds()
cats = coco.loadCats(catIDs)


def getClassName(classID: int, cats_: list):
    for i in range(len(cats_)):
        if cats_[i]['id'] == classID:
            return cats_[i]['name']
    return "None"


print('The class name is', getClassName(1, cats))

# Define the classes (out of the 81) which you want to see. Others will not be shown.
filterClasses = ['white_cube', 'purple_cube', 'blue_cube']

# Fetch class IDs only corresponding to the filterClasses
catIds = coco.getCatIds(catNms=filterClasses)
# Get all images containing the above Category IDs
imgIds = coco.getImgIds(catIds=catIds)
print("Number of images containing all the  classes:", len(imgIds))

# load and display a random image
img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
I = io.imread(f'{dataDir}/{img["file_name"]}')/255.0
n_plots = 3
plt.subplot(1,n_plots,1)
plt.axis('off')
plt.imshow(I)


annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)

def allCombinationsFilter(classes):
    images = []
    if classes != None:
        # iterate for each individual class in the list
        for className in classes:
            # get all images containing given class
            catIds = coco.getCatIds(catNms=className)
            imgIds = coco.getImgIds(catIds=catIds)
            images += coco.loadImgs(imgIds)
    else:
        imgIds = coco.getImgIds()
        images = coco.loadImgs(imgIds)

    # Now, filter out the repeated images
    unique_images = []
    for i in range(len(images)):
        if images[i] not in unique_images:
            unique_images.append(images[i])

    dataset_size = len(unique_images)
    return dataset_size


print("Number of images containing the filter classes:", allCombinationsFilter(filterClasses))

plt.subplot(1,n_plots,2)
plt.axis('off')
#### GENERATE A SEGMENTATION MASK ####
mask = np.zeros((img['height'],img['width']))
for i in range(len(anns)):
    className = getClassName(anns[i]['category_id'], cats)
    pixel_value = filterClasses.index(className)+1
    mask = np.maximum(coco.annToMask(anns[i])*pixel_value, mask)
plt.imshow(mask)


plt.subplot(1,n_plots,3)
plt.axis('off')
#### GENERATE A BINARY MASK ####
mask = np.zeros((img['height'],img['width']))
for i in range(len(anns)):
    mask = np.maximum(coco.annToMask(anns[i]), mask)
plt.imshow(mask)
plt.show()
'''


def filterDataset(folder:str, classes:list, validation_count: int=10):
    # initialize COCO api for instance annotations
    annFile = fr'{folder}/result.json'
    coco = COCO(annFile)

    images = []

    if classes != None:
        # iterate for each individual class in the list
        for className in classes:
            # get all images containing given categories
            catIds = coco.getCatIds(catNms=className)
            imgIds = coco.getImgIds(catIds=catIds)
            images += coco.loadImgs(imgIds)
    else:
        imgIds = coco.getImgIds()
        images = coco.loadImgs(imgIds)

    unique_images = []
    for i in range(len(images)):
        if images[i] not in unique_images:
            unique_images.append(images[i])

    random.shuffle(unique_images)
    dataset_size = len(unique_images)

    return unique_images[validation_count:], unique_images[:validation_count], dataset_size, coco


def add_noise(img):
    mean = 0
    stddev = 180
    noise = np.zeros(img.shape, np.uint8)
    cv2.randn(noise, mean, stddev)

    return cv2.add(img, noise)

def get_dataset(dir):
    masks = []
    images = []
    files_labels = listdir(dir+'/labeled')
    for i, file in enumerate(listdir(dir+'/raw')):
        if 'png' in file:
            image = cv2.imread(dir+'/raw'+'/'+file, cv2.IMREAD_GRAYSCALE)
            blur_img = cv2.GaussianBlur(image, (3, 3), 0)
            noise = add_noise(image)
            blur_noise = add_noise(blur_img)
            images.extend([image, blur_img, noise, blur_noise])
            if f'{i+1}.png' in files_labels:
                masks.extend([cv2.imread(dir+'/labeled'+'/'+f'{i+1}.png', cv2.IMREAD_GRAYSCALE) for k in range(4)])
            else:
                masks.extend([np.zeros(input_image_size) for k in range(4)])
    return images, masks

def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == classID:
            return cats[i]['name']
    return None


def getImage(imageObj, img_folder, input_image_size):
    # Read and normalize an image
    train_img = io.imread(img_folder + '/' + imageObj['file_name']) / 255.0
    # Resize
    train_img = cv2.resize(train_img, input_image_size)

    if (len(train_img.shape) == 3 and train_img.shape[2] == 3):  # If it is a RGB 3 channel image
        return train_img
    else:  # To handle a black and white image, increase dimensions to 3
        stacked_img = np.stack((train_img,) * 3, axis=-1)
        return stacked_img


def getNormalMask(imageObj, classes, coco, catIds, input_image_size):
    annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    cats = coco.loadCats(catIds)
    train_mask = np.zeros(input_image_size)

    for a in range(len(anns)):
        className = getClassName(anns[a]['category_id'], cats)
        pixel_value = classes.index(className) + 1
        new_mask = cv2.resize(coco.annToMask(anns[a]) * pixel_value, input_image_size)
        train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
    return train_mask


def getBinaryMask(imageObj, coco, catIds, input_image_size):
    annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    train_mask = np.zeros(input_image_size)
    for a in range(len(anns)):
        new_mask = cv2.resize(coco.annToMask(anns[a]), input_image_size)

        # Threshold because resizing may cause extraneous values
        new_mask[new_mask >= 0.5] = 1
        new_mask[new_mask < 0.5] = 0

        train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
    return train_mask


def dataGeneratorCoco(images, classes, coco, folder, input_image_size=(512, 512), mask_type='binary'):
    img_folder = folder
    dataset_size = len(images)
    catIds = coco.getCatIds(catNms=classes)

    img = np.zeros((dataset_size, input_image_size[0], input_image_size[1], 3)).astype('float')
    mask = np.zeros((dataset_size, input_image_size[0], input_image_size[1], 1)).astype('float')

    for i in range(dataset_size):
        imageObj = images[i]

        ### Retrieve Image ###
        train_img = getImage(imageObj, img_folder, input_image_size)

        ### Create Mask ###
        if mask_type == "binary":
            train_mask = getBinaryMask(imageObj, coco, catIds, input_image_size)

        elif mask_type == "normal":
            train_mask = getNormalMask(imageObj, classes, coco, catIds, input_image_size)

            # Add to respective batch sized arrays
        img[i] = train_img
        mask[i] = train_mask
    return img, mask



def visualizeGenerator(gen):
    img, mask = gen[0]

    fig = plt.figure(figsize=(20, 10))
    outerGrid = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1)

    for i in range(2):
        innerGrid = gridspec.GridSpecFromSubplotSpec(2, 2,
                                                     subplot_spec=outerGrid[i], wspace=0.05, hspace=0.05)

        for j in range(4):
            ax = plt.Subplot(fig, innerGrid[j])
            if (i == 1):
                ax.imshow(img[j])
            else:
                ax.imshow(mask[j][:, :, 0])

            ax.axis('off')
            fig.add_subplot(ax)
    plt.show()


def augmentationsGenerator(gen, augGeneratorArgs=None, seed=101):
    (img, mask) = gen
    dataset_size = len(img)
    imgs = np.zeros((dataset_size, input_image_size[0], input_image_size[1], 3)).astype('float')
    masks = np.zeros((dataset_size, input_image_size[0], input_image_size[1], 1)).astype('float')

    image_gen = ImageDataGenerator(**augGeneratorArgs)
    augGeneratorArgs_mask = augGeneratorArgs.copy()
    _ = augGeneratorArgs_mask.pop('brightness_range', None)
    mask_gen = ImageDataGenerator(**augGeneratorArgs_mask)

    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))

    for i in range(len(img)):
        seed = np.random.choice(range(9999))
        g_x = image_gen.flow(255 * np.expand_dims(img[i], axis=0),
                             batch_size=1,
                             seed=seed,
                             shuffle=False)
        g_y = mask_gen.flow(np.expand_dims(mask[i], axis=0),
                            batch_size=1,
                            seed=seed,
                            shuffle=False)

        imgs[i] = next(g_x) / 255.0
        masks[i] = next(g_y)
    return imgs, masks


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (1,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


def get_mask(pred, i=0):
    mask = np.argmax(pred[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = mask*(255/(num_classes-1))
    return img


def get_colored_mask(mask, num_classes, colors, classes_names=None,threshold=5):
    classes_masks = np.ndarray((num_classes-1,512,512))

    for i in range(num_classes-1):
        lower = int(255 / (num_classes - 1) * (i + 1) - threshold)
        higher = int(255 / (num_classes - 1) * (i + 1) + threshold)
        class_mask = cv2.inRange(mask,lower,higher if higher <= 255 else 255)
        classes_masks[i] = class_mask
    classes_masks_color = np.ndarray((num_classes-1,512,512,3)).astype(np.uint8)

    for i, m in enumerate(classes_masks):
        colored_image = np.zeros((m.shape[0], m.shape[1], 3))
        colored_image[:,:,0] = cv2.cvtColor(np.float32(np.asarray(colors[i]).reshape((1,1,3))), cv2.COLOR_RGB2HSV)[0][0][0]
        colored_image[:,:,1] = cv2.cvtColor(np.float32(np.asarray(colors[i]).reshape((1,1,3))), cv2.COLOR_RGB2HSV)[0][0][1]
        colored_image[:, :, 2] = m
        colored_image = cv2.cvtColor(np.float32(colored_image), cv2.COLOR_HSV2BGR)
        classes_masks_color[i] = colored_image

    final_mask = np.sum(classes_masks_color, axis=0)
    final_mask = (final_mask*255).astype(np.uint8)

    if np.max(final_mask) > 255:
        final_mask = np.clip(final_mask, 0, 255)
    return final_mask


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


def get_masked_img(image):
    val_preds = model.predict(cv2.resize(image, input_image_size).reshape(1, *input_image_size, 3), verbose=0)
    mask_gen = get_mask(val_preds)
    mask_gen_overlay = np.asarray(mask_gen)
    masked_image = get_colored_mask(mask_gen_overlay, num_classes, classes_colors, filterClasses)
    masked_image = cv2.addWeighted(image, 1, masked_image, 0.9, 0, dtype = cv2.CV_32F)
    return masked_image


def center_crop(im):
    center = im.shape
    min_dim = min(center[:2])
    x = center[1] / 2 - min_dim / 2
    y = center[0] / 2 - min_dim / 2
    return im[int(y):int(y+min_dim), int(x):int(x+min_dim)]


def dice_metric(y_true, y_pred):
    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
    union = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))
    '''
    if y_pred.sum() == 0 and y_pred.sum() == 0:
        return 1.0
    '''

    return 2*intersection / union


def jaccard_distance_loss(x_true, x_pred, smooth=100):
    intersection = K.sum(K.sum(K.abs(x_true * x_pred), axis=-1))
    sum_ = K.sum(K.sum(K.abs(x_true) + K.abs(x_pred), axis=-1))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def IoU(x_true, x_pred):
    results = []
    print(x_true)
    #if x_true.shape[0] is None:return 0.
    for i in range(0, x_true.shape[1]):

        # boxTrue
        x_boxTrue_tleft = x_true[0, 0]  # numpy index selection
        y_boxTrue_tleft = x_true[0, 1]
        boxTrue_width = x_true[0, 2]
        boxTrue_height = x_true[0, 3]
        area_boxTrue = (boxTrue_width * boxTrue_height)

        # boxPred
        x_boxPred_tleft = x_pred[0, 0]
        y_boxPred_tleft = x_pred[0, 1]
        boxPred_width = x_pred[0, 2]
        boxPred_height = x_pred[0, 3]
        area_boxPred = (boxPred_width * boxPred_height)

        # calculate the bottom right coordinates for boxTrue and boxPred

        # boxTrue
        x_boxTrue_br = x_boxTrue_tleft + boxTrue_width
        y_boxTrue_br = y_boxTrue_tleft + boxTrue_height  # Version 2 revision

        # boxPred
        x_boxPred_br = x_boxPred_tleft + boxPred_width
        y_boxPred_br = y_boxPred_tleft + boxPred_height  # Version 2 revision

        # calculate the top left and bottom right coordinates for the intersection box, boxInt

        # boxInt - top left coords
        x_boxInt_tleft = np.max([x_boxTrue_tleft, x_boxPred_tleft])
        y_boxInt_tleft = np.max([y_boxTrue_tleft, y_boxPred_tleft])  # Version 2 revision

        # boxInt - bottom right coords
        x_boxInt_br = np.min([x_boxTrue_br, x_boxPred_br])
        y_boxInt_br = np.min([y_boxTrue_br, y_boxPred_br])

        # Calculate the area of boxInt, i.e. the area of the intersection
        # between boxTrue and boxPred.
        # The np.max() function forces the intersection area to 0 if the boxes don't overlap.

        # Version 2 revision
        area_of_intersection = \
            np.max([0, (x_boxInt_br - x_boxInt_tleft)]) * np.max([0, (y_boxInt_br - y_boxInt_tleft)])

        iou = area_of_intersection / ((area_boxTrue + area_boxPred) - area_of_intersection)

        # append the result to a list at the end of each loop
        results.append(iou)

    # return the mean IoU score for the batch
    return np.mean(results)


def retrain(optimizer, loss, metrics, train, val):
    i = 0
    while os.path.exists(f"weights/coco_scnn_E{n_epochs}_v{i+1}.h5"):
        i += 1
    print(f'load model: coco_scnn_E{n_epochs}_v{i}.h5')

    model_local = keras.models.load_model(f'weights/coco_scnn_E{n_epochs}_v{i}.h5')

    x = model_local.layers[-2].output
    x = layers.Conv2D(num_classes, kernel_size=1, activation='softmax')(x)
    new_model = keras.models.Model(inputs=model_local.input, outputs=x)

    for layer in new_model.layers[:-1]:
        layer.trainable = False

    new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    history = new_model.fit(x=train,
                            epochs=n_epochs,
                            validation_data=val,
                            steps_per_epoch=steps_per_epoch,
                            validation_steps=validation_steps)

    i = 0
    while os.path.exists(f"weights/retrain_coco_scnn_E{n_epochs}_v{i}.h5"):
        i += 1
    new_model.save(f'weights/retrain_coco_scnn_E{n_epochs}_v{i}.h5')
    return new_model, history


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Hyperparameters
keras.utils.set_random_seed(101)
n_epochs = 10
batch_size = 4
num_classes = 2
val_count = 1
len_of_dataset = 164
steps_per_epoch = (len_of_dataset - 0) // batch_size
validation_steps = val_count // batch_size
input_image_size = (512, 512)
mask_type = 'normal'
dataDir = r'data/dataset/'
filterClasses = ['mark']
classes_colors = [[0, 100, 0]]
augGeneratorArgs = dict(featurewise_center = False,
                        samplewise_center = False,
                        rotation_range = 0,
                        width_shift_range = 0.0,
                        height_shift_range = 0.0,
                        brightness_range = (0.8,1.2),
                        shear_range = 0.0,
                        zoom_range = [1, 1],
                        horizontal_flip = False,
                        vertical_flip = False,
                        #fill_mode = 'reflect',
                        data_format = 'channels_last')

# Generate train and validation arrays
#train, val, dataset_size, coco = filterDataset(dataDir, filterClasses, validation_count=val_count)
#train_gen = dataGeneratorCoco(train, filterClasses, coco, dataDir, input_image_size, mask_type)
#val_gen = dataGeneratorCoco(val, filterClasses, coco, dataDir, input_image_size, mask_type)
#train_x, train_y = augmentationsGenerator(train_gen, augGeneratorArgs)
#val_x, val_y = augmentationsGenerator(val_gen, augGeneratorArgs)
train_x, train_y = np.asarray(get_dataset(dataDir))
val_x, val_y = np.asarray([cv2.imread(dataDir+'/'+'Screenshot_5_11.png', cv2.IMREAD_GRAYSCALE)]), \
                np.asarray([cv2.imread(dataDir+'/'+'36.png', cv2.IMREAD_GRAYSCALE)])
img_index = 0
images_number = 0
model = None
load_model = 1
retrain_model = 0
plot_architecture = 1

if not load_model:
    img_index = 1
    images_number = 4
    # Build model
    model = get_model(input_image_size, num_classes)
    optimizer = "rmsprop"
    loss = "sparse_categorical_crossentropy"
    metrics = [
        'accuracy',
        #dice_metric,
        #jaccard_distance_loss,
    ]
    plt.subplot(1, images_number, 1)
    history = None
    i = 0
    if retrain_model:
        model, history = retrain(optimizer, loss, metrics, (train_x, train_y), (val_x, val_y))
    else:
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        # Start training
        history = model.fit(train_x,
                            train_y,
                            validation_data=(val_x, val_y),
                            batch_size=batch_size,
                            epochs = n_epochs,
                            verbose = True)

    if not retrain_model:
        i = 0
        while os.path.exists(f"weights/coco_scnn_E{n_epochs}_v{i}.h5"):
            i += 1
        model.save(f'weights/coco_scnn_E{n_epochs}_v{i}.h5')

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{f"coco_scnn_E{n_epochs}_v{i}"} fitting history; Metric:MeanIoU')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
else:
    if retrain_model:
        images_number = 3
        i = 0
        while os.path.exists(f"weights/retrain_coco_scnn_E{n_epochs}_v{i+1}.h5"):
            i += 1
        print(f'load model: retrain_coco_scnn_E{n_epochs}_v{i}.h5')
        model = keras.models.load_model(f'weights/retrain_coco_scnn_E{n_epochs}_v{i}.h5')
    else:
        images_number = 3
        i = 0
        while os.path.exists(f"weights/coco_scnn_E{n_epochs}_v{i+1}.h5"):
            i += 1
        print(f'load model: coco_scnn_E{n_epochs}_v{i}.h5')
        model = keras.models.load_model(f'weights/coco_scnn_E{n_epochs}_v{i}.h5')





image_gen = cv2.imread(dataDir+'/'+'Screenshot_5_11.png', cv2.IMREAD_GRAYSCALE)
image = image_gen
plt.subplot(1, images_number, img_index+1)
plt.imshow(image)
val_preds = model.predict(cv2.resize(image, input_image_size).reshape(1,*input_image_size,1))

plt.subplot(1, images_number, img_index+2)
mask_gen = get_mask(val_preds)
plt.imshow(mask_gen)

plt.subplot(1, images_number, img_index+3)
mask_gen_overlay = np.asarray(mask_gen)
masked_image = get_colored_mask(mask_gen_overlay, num_classes, classes_colors, filterClasses)
masked_image = cv2.addWeighted(image, 1, masked_image, 0.9, 0, dtype = cv2.CV_32F)
plt.imshow(masked_image)
plt.show()

cap = cv2.VideoCapture(0)

while (True):
    st = time.time()
    ret, frame = cap.read()
    frame = center_crop(frame)
    frame = image_resize(frame, width=input_image_size[0], height=input_image_size[1])
    frame = get_masked_img(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255)
    cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f'frame gen time: {(time.time() - st):.3f}s')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()




