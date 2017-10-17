import numpy as np
import cv2
import os
from sklearn import preprocessing
from skimage import exposure
import random

random.seed(1769)
DataValidationFolder =['01']

def load_names(val_seq = -1, augm=0):

    gt_dir = 'C:\\Users\Diego\Desktop\FaceExtraction\Extraction\\'
    if val_seq < 0:
        # load all and remove validation sequence
        gt_list = os.listdir(gt_dir)
        for el in DataValidationFolder:
            to_remove = os.path.join(el)
            gt_list.remove(to_remove)
    else:
        # load validation sequence
        gt_list = DataValidationFolder
    data_face = []
    data_noface = []
    nofaceIndex=0
    for gt_folder in gt_list:
        for gt_trial_folder in os.listdir(os.path.join(gt_dir,gt_folder)):
            for gt_file_face in os.listdir(os.path.join(gt_dir,gt_folder,gt_trial_folder,'DEPTH','face')):
                img_name= os.path.join(gt_dir,gt_folder,gt_trial_folder,'DEPTH','face',gt_file_face)
                data_face.append({'image': img_name,'face':True,'augm': int(augm)})
            for gt_file_noface in os.listdir(os.path.join(gt_dir,gt_folder, gt_trial_folder, 'DEPTH', 'noface')):
                img_name = os.path.join(gt_dir,gt_folder, gt_trial_folder, 'DEPTH', 'noface', gt_file_noface)
                data_noface.append({'image': img_name, 'face':False,'augm': int(augm)})
                if nofaceIndex%9==0:
                    nofaceIndex=0
    # balancing data (face - no face)
    random.shuffle(data_noface)
    data_noface=data_noface[:len(data_face)]
    data = data_face + data_noface
    return data


def load_names_val():
    return load_names(val_seq=1, augm=0)


def identity(img):
    return img


def Flip(img):
    return cv2.flip(img, 1)


def Traleft(img):
    M = np.float32([[1, 0, -(img.shape[0]/4)], [0, 1, 0]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def Traright(img):
    M = np.float32([[1, 0, (img.shape[0] / 4)], [0, 1, 0]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def Traup(img):
    M = np.float32([[1, 0, 0], [0, 1, -(img.shape[0] / 4)]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def Tradown(img):
    M = np.float32([[1, 0, 0], [0, 1, (img.shape[0]/4 )]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def Traleftup(img):
    M = np.float32([[1, 0, -(img.shape[0] / 4)], [0, 1, -(img.shape[0] / 4)]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def Trarightup(img):
    M = np.float32([[1, 0, (img.shape[0] / 4)], [0, 1, -(img.shape[0] / 4)]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def Traleftdown(img):
    M = np.float32([[1, 0, -(img.shape[0] / 4)], [0, 1, (img.shape[0] / 4)]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def Trarightdown(img):
    M = np.float32([[1, 0, (img.shape[0] / 4)], [0, 1, (img.shape[0] / 4)]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

# map the inputs to the function blocks
Augmentation = {
    0: identity,
    1 : Tradown,
    2 : Traup,
    3 : Traleft,
    4 : Traright,
    5 : Flip,
    6 : Traleftdown,
    7 : Traleftup,
    8 : Trarightdown,
    9 : Trarightup,
}


def load_images(train_data_names, crop, scale, rescale, normcv2, b_debug,fulldepth, rows, cols,equalize,removeBackground):

    # channel
    ch = 1
    # image dimensions
    rows = rows
    cols = cols

    # boolean for visualization
    b_debug = b_debug

    # data structure for image
    img_batch = np.zeros(shape=(len(train_data_names), ch, rows, cols), dtype=np.float32)
    # data structure for GT
    y_batch = np.zeros(shape=(len(train_data_names), 2), dtype=np.float32)
    # fill structure
    for i, line in enumerate(train_data_names):
        # image name
        img_name = line['image']
        img = cv2.imread(img_name, cv2.IMREAD_ANYDEPTH)

        if removeBackground:
            center = np.mean(img[int(img.shape[0] / 2 - 1):int(img.shape[0] / 2 + 2), int(img.shape[1] / 2 - 1):int(img.shape[1] / 2 + 2)])
            img[img > center + 150] = 0

        if fulldepth:
            img = (img - 500) / 8.0
            img = img.astype(np.uint8)

        if crop:
            cropsize = 15
            img = img[cropsize:img.shape[1]-cropsize+1, cropsize:img.shape[0]-cropsize+1]

        # data augmentation
        img = Augmentation[line['augm']](img)

        if equalize:
            if fulldepth:
                img = cv2.equalizeHist(img)
        # debug
        if b_debug:
            if line['augm'] != 0:
                cv2.imshow("caricanda augm:{}".format(line['augm']), cv2.resize((img).astype(np.uint8) * 255, (0, 0), fx=5, fy=5))
                cv2.waitKey()

        # Rescale
        if rescale:
            img = exposure.rescale_intensity(img.astype('float'), in_range=(500, 4500), out_range=(0, 1))

        # Scale
        if scale:
            img = preprocessing.scale(img.astype('float'))

        # Normalize (openCV)
        if normcv2:
            cv2.normalize(img.astype('float'), img, alpha=-1.0, beta=1.0, norm_type=cv2.NORM_MINMAX)

        # resize
        img = cv2.resize(img, (cols, rows))

        # add channel dimension
        img = np.expand_dims(img, 2)

        img = img.astype(np.float32)

        # load batch
        img_batch[i] = img.transpose(2, 0, 1)

        # load gt
        y_batch[i] = [int(line['face']),int(not(line['face']))]
    return img_batch, y_batch





