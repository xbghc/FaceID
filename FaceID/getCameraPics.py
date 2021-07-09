# 从摄像头拍摄照片

import os
import cv2


def cameraAutoForPictures(user='anonymous'):
    saveDir = '../data/{}/'.format(user)
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    count = 1
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    width, height, w = 640, 480, 360
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    crop_w_start = (width - w) // 2
    crop_h_start = (height - w) // 2
    print('width: ', width)
    print('height: ', height)

    while True:
        ret, frame = cap.read()
        frame = frame[crop_h_start:crop_h_start + w, crop_w_start:crop_w_start + w]
        frame = cv2.flip(frame, 1, dst=None)
        cv2.imshow("capture", frame)
        action = cv2.waitKey(1) & 0xFF
        if action == ord('p'):
            cv2.imwrite(f"{saveDir}/{count}.jpg", cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA))
            print(f"{saveDir}:{count}张图片")
            count += 1
        elif action == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # username = input("请输入姓名(不支持中文):")
    username = 'cx'
    cameraAutoForPictures(user=username)
