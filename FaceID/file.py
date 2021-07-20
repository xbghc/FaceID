import os
import base64
import cv2
import numpy as np


def endwith(s, *endstring):
    resultArray = map(s.endswith, endstring)
    if True in resultArray:
        return True
    return False


def read_file(path):
    img_list = []
    label_list = []
    dir_counter = 0
    IMG_SIZE = 128
    for child_dir in os.listdir(path):
        child_path = os.path.join(path, child_dir)
        for dir_image in os.listdir(child_path):
            if endwith(dir_image, 'jpg'):
                img = cv2.imread(os.path.join(child_path, dir_image))
                resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                recolored_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
                img_list.append(recolored_img)
                label_list.append(dir_counter)
        dir_counter += 1
    img_list = np.array(img_list)
    return img_list, label_list, dir_counter


def read_name_list(path):
    name_list = []
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
    return name_list


def readAllImg(path, *suffix):
    try:
        s = os.listdir(path)
        resultArray = []
        fileName = os.path.basename(path)
        resultArray.append(fileName)
        for i in s:
            if endwith(i, suffix):
                document = os.path.join(path, i)
                img = cv2.imread(document)
                resultArray.append(img)
    except IOError:
        print("Error")

    else:
        print("读取成功")
        return resultArray



