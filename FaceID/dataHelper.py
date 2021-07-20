import os
import cv2
import time

from FaceID.file import readAllImg


def readPicSaveFace(sourcePath, objectPath, *suffix):
    if not os.path.exists(objectPath):
        os.makedirs(objectPath)
    try:
        resultArray = readAllImg(sourcePath, *suffix)
        count = 1
        face_cascade = cv2.CascadeClassifier('../docs/haarcascade_frontalface_alt.xml')
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
    # username = input("用户名：")
    username = 'ghm'
    dataDir = f'../data/{username}/'
    datasetDir = f'../dataset/{username}/'
    print(f'源文件夹：{dataDir}\n目的文件夹：{datasetDir}')
    readPicSaveFace(dataDir, datasetDir, '.jpg', '.JPG', 'png', 'PNG', 'tiff')

