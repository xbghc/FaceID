import os
import cv2
import time


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


def endwith(s, *endstring):
    resultArray = map(s.endswith, endstring)
    if True in resultArray:
        return True
    else:
        return False


def readPicSaveFace(sourcePath, objectPath, *suffix):
    if not os.path.exists(objectPath):
        os.makedirs(objectPath)
        try:
            resultArray = readAllImg(sourcePath, *suffix)
            count = 1
            face_cascade = cv2.CascadeClassifier('config/haarcascade_frontalface_alt.xml')
            for i in resultArray:
                if type(i) != str:
                    gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    for (x, y, w, h) in faces:
                        listStr = [str(int(time.time())), str(count)]
                        fileName = ''.join(listStr)
                        f = cv2.resize(gray[y:(y + h), x:(x + w)], (200, 200))
                        cv2.imwrite(objectPath + os.sep + '%s.jpg' % fileName, f)
                        count += 1
        except Exception as e:
            print("Exception:", e)
        else:
            print('Read ' + str(count - 1) + ' Faces to Destination ' + objectPath)


if __name__ == '__main__':
    print('dataProcessing!!!')
    readPicSaveFace('data/guanxijing/', 'dataset/guanxijing/', ' .jpg', '.JPG', 'png', 'PNG', 'tiff')

readPicSaveFace('data/KA/', 'dataset/KA/', '.jpg', '.JPG', 'png', 'PNG', 'tiff')