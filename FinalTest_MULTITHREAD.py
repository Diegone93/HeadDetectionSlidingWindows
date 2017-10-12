import sys
import pandas as pd
import re
import os
from math import sqrt
import time

if len(sys.argv) > 20:
    gpu_id = sys.argv[1]
else:
    gpu_id = "gpu"
    cnmem = "0.10"
print("Argument: gpu={}".format(gpu_id))
os.environ["THEANO_FLAGS"] = "device=" + gpu_id + ", lib.cnmem=" + cnmem

from models import faceNet2,faceNet3
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")


def checkBadFrame(sequence, frame):

    if sequence == 'data_04-24-38':
        if frame in range(32, 64) or frame in range(268, 273):
            return True

    if sequence == 'data_03-54-47':
        if frame in range(1, 8) or frame in range(14, 23):
            return True

    if sequence == 'data_04-21-34':
        if frame in range(70, 84):
            return True

    if sequence == 'data_04-20-27':
        if frame in range(165, 168):
            return True

    if sequence == 'data_04-15-01':
        if frame in range(108, 110):
            return True

    if sequence == 'data_03-31-37':
        if frame in [149, 151, 39, 40, 50, 30, 31, 36, 37, 38]:
            return True

    if sequence == 'data_04-22-13':
        if frame in range(76, 86) or frame in range(135, 140):
            return True

    if sequence == 'data_03-31-52':
        if frame in range(89, 103):
            return True

    if sequence == 'data_04-17-41':
        if frame in range(76, 85):
            return True

    if sequence == 'data_04-21-43':
        if frame in range(100, 105) or frame in range(57, 81) or frame in range(21, 34):
            return True

    if sequence == 'data_04-36-55':
        if frame in range(130, 137):
            return True

    if sequence == 'data_04-34-45':
        if frame in range(191, 194):
            return True

    if sequence == 'data_04-25-52':
        if frame in range(130, 137):
            return True

    if sequence == 'data_04-21-26':
        if frame in range(146, 150):
            return True

    return False



def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    #print xA,yA,xB,yB
    # compute the area of intersection rectangle
    if (xB - xA ) < 0 or (yB - yA )<0:
        return 0
    interArea = (xB - xA ) * (yB - yA )

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def getWindows(img,kernel_dim=11):
    center = []
    x, y = int((kernel_dim - 1) / 2), int((kernel_dim - 1) / 2)

    while True:
        F = 365.337
        R = 245.0
        minus = (kernel_dim - 1) / 2
        plus = ((kernel_dim) / 2) + 1
        box_Range = img[y - minus:y + plus, x - minus :x + plus]
        box_mean = (np.mean(box_Range))
        if box_mean < 700:
            box_mean = 10000
        Range = int(round(F * R / float(box_mean)))
        if ((x - Range / 2) >= 0) and (y - Range / 2) >= 0 and ((x + Range / 2) <= img.shape[1]) and ((y + Range / 2) <= img.shape[0]) :
            center.append({'x': int(x), 'y': int(y), 'range': int(Range), 'value_center': int(box_mean)})

        x = x + int((kernel_dim+1) / 2)
        if x > img.shape[1]:
            x = int((kernel_dim - 1) / 2)
            y = y + int((kernel_dim+1) / 2)
        if y > img.shape[0]:
            break

    centers = np.array([each.values() for each in center])
    windows = np.ones([centers.shape[0], 1, 64, 64]) * 0

    # X -> [1] Y -> [0]

    for i, w in enumerate(centers):
        # crop bounding box
        im = img[w[0] - (w[2]/2):w[0] + (w[2]/2) + 1, w[1] - w[2]/2:w[1] + (w[2]/2) + 1].copy()

        # remove background
        head_size = 250
        im[im > (w[3] + head_size)] = 0

        im = cv2.normalize(im.astype('float'), alpha=-1.0, beta=1.0, norm_type=cv2.NORM_MINMAX)

        im = cv2.resize(im, (64, 64))
        im = np.expand_dims(im, 2)
        im = im.astype(np.float32)
        windows[i, :, :] = (im.transpose(2, 0, 1))

    return centers,windows

def prediction(filename,model,skel,imgnum,detection,localization,save=False,show=True):
    head = False
    kinectHead = False
    localized = False
    centerDistance = 0.0
    iou = 0.0

    img = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)

    if checkBadFrame(filename.split('\\')[-3],imgnum):
        return -1, kinectHead, head, localized, iou, centerDistance

    if skel[2] > 0:
        try:
            x = int(round(skel[9]))
            y = int(round(skel[10]))
            kinectHead = True
            F = 365.337
            R = 245.0
            GT_kernel_size = 11
            minus = (GT_kernel_size - 1) / 2
            plus = ((GT_kernel_size) / 2) + 1
            box_Range = img[y - minus:y + plus, x - minus:x + plus]
            box_mean = (np.mean(box_Range))
            range = int(round(F * R / float(box_mean)))
            if range > 100:
                print 'ERRORE RANGE {} GT {}'.format(range, imgnum)
                return -1, kinectHead, head, localized, iou, centerDistance

        except:
            kinectHead = False
            print 'ERRORE ESTRAZIONE GT {}'.format(imgnum)
            return -1, kinectHead, head, localized, iou, centerDistance
    else:
        print 'GT NOT FOUND {}'.format(imgnum)
        return -1, kinectHead, head, localized, iou, centerDistance

    t = time.time()
    centers, win = getWindows(img)
    # prediction
    pred = model.predict(x=win, batch_size=batch_size, verbose=0)
    max = np.argmax(pred[:, 0])
    t = time.time() - t

    img = (img - 500) / 8.0
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    if kinectHead :
        cv2.rectangle(img, (x - range / 2, y - range/ 2),(x + range / 2, y + range / 2),(0, 0, 255), 1)

    if show or save:
        for c in centers[pred[:,0] > 0.95]:
            cv2.rectangle(img, (c[1] - c[2] / 2, c[0] - c[2] / 2),(c[1] + c[2] / 2, c[0] + c[2] / 2), (0, 255, 0), 1)

    if np.max((pred[:, 0]))>0.95:
        head=True
        if show or save:
            cv2.circle(img, (centers[max][1], centers[max][0]), 2, (255, 0, 0), -1)
            cv2.rectangle(img, (centers[max][1] - centers[max][2] / 2, centers[max][0] - centers[max][2] / 2),(centers[max][1] + centers[max][2] / 2, centers[max][0] + centers[max][2] / 2), (255, 0, 0), 1)

    if kinectHead and head:
        iou = bb_intersection_over_union(
                                            [centers[max][1] - centers[max][2] / 2,centers[max][0] - centers[max][2] / 2,centers[max][1] + centers[max][2] / 2,centers[max][0] + centers[max][2] / 2],
                                            [x - range / 2, y - range/ 2,x + range / 2, y + range / 2]
                                        )
        centerDistance = sqrt((centers[max][1]-x)**2+(centers[max][0]-y)**2)

        if iou >= 0.3:
            localized = True

        cv2.putText(img, 'head: {0} localized: {1} FPS: {2:.2f} IoU: {3:.2f} D: {4:.2f}'.format(head,localized,1.0/t, iou, centerDistance), (15, 15), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
    else:
        cv2.putText(img, 'head: {0} localized: {1} FPS: {2:.2f}'.format(head,localized,1.0/t), (15, 15), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)


    if show:
        cv2.imshow('', cv2.resize(img,(0,0),fx=2,fy=2))
        cv2.waitKey(1)


    if head and iou >= 0.3:
        detection['TP'] += 1
    else:
        detection['FP'] += 1

    if save:
        try:
            os.mkdir('D:\Diego\FaceExtraction\depthFaceDetection\FinalTestScoreImage\{}'.format(filename.split('\\')[-3]))
        except:
            pass
        cv2.imwrite('D:\Diego\FaceExtraction\depthFaceDetection\FinalTestScoreImage\{}\{}'.format(filename.split('\\')[-3],filename.split('\\')[-1]),img)

    return 1.0/t,kinectHead,head,localized,iou,centerDistance

if __name__ == '__main__':

    # image parameters
    rows = 64
    cols = 64
    ch = 1

    # training parameters
    b_crop = False
    b_rescale = False
    b_scale = False
    b_normcv2 = True

    batch_size = 400

    # image visualization
    b_visualize = False

    # graph visualization
    b_plot = True

    # model
    model = faceNet2(rows, cols)
    #model = faceNet3(rows,cols)
    # weights
    print("Load weights ...")
    model.load_weights('weights_5_16bit\weights.013-0.03280.hdf5')
    print("Done.")
    soggettitest = ['01', '15']

    trials = ['data_01-05-11',
              'data_01-11-11',
              'data_01-12-07',
              'data_01-13-44',
              'data_03-27-38',
              'data_03-31-37',
              'data_03-31-52',
              'data_03-54-47',
              'data_04-14-58',
              'data_04-15-01',
              'data_04-17-41',
              'data_04-20-27',
              'data_04-21-26',
              'data_04-21-34',
              'data_04-21-43',
              'data_04-22-13',
              'data_04-24-38',
              'data_04-25-52',
              'data_04-34-45',
              'data_04-36-55']

    for trial in trials:
        path = 'C:\Users\Diego\Desktop\Dataset\Dataset_test\confronto\\' + trial
        print '\nworking on {}'.format(os.path.join(path))
        with open(os.path.join(path, 'data.txt')) as f:
            lines = f.readlines()
        OUT = []
        detection = {'TP': 0, 'FP': 0}
        localization = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        for index,img in enumerate(sorted(os.listdir(os.path.join(path, 'DEPTH')), key=lambda x: (int(re.sub('\D','',x)),x))):
            skel = np.fromstring(lines[index], sep='\t')
            fps, gt_head, head, localized, iou, distance = prediction(os.path.join(path, 'DEPTH',img),model,skel,index+1,detection=detection,localization=localization,save=False,show=True)
            if fps < 0 :
                print 'FRAME SKIPPED'
                continue
            sys.stdout.write("\r%.2f%%" % (((index+1) / float(len(os.listdir(os.path.join(path, 'DEPTH'))))) * 100))
            sys.stdout.flush()
            DICT = {'FPS': '', 'HEAD_GT': '', 'DETECTION': '', 'LOCALIZATION': '', 'IOU': 0,'CENTERDISTANCE': 0}
            DICT['FPS']=fps
            DICT['HEAD_GT']=gt_head
            DICT['DETECTION']=head
            DICT['LOCALIZATION']=localized
            DICT['IOU']=iou
            DICT['CENTERDISTANCE']=distance
            OUT.append(DICT)

        out = pd.DataFrame(data=OUT)
        out.to_csv(os.path.join(path, 'FRAMEXFRAMEPREDICTION_k11.csv'))
        detectionDF = pd.DataFrame(data=[detection])
        detectionDF.to_csv(os.path.join(path, 'DETECTION_k11.csv'))
