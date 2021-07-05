import os
import cv2
from faceRegnigtionModel import Model

threshold = 0.7


def read_name_list(path):
    name_list = []
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
    return name_list


class Camera_reader(object):
    def _init_(self):
        self.model = Model()
        self.model.load()
        self.img_size = 128

    def build_camera(self):
        face_cascade = cv2.CascadeClassifier('config/haarcascade_frontalface+alt.xml')
        name_list = read_name_list('dataset/')
        cameraCapture = cv2.VideoCapture(0)
        success, frame = cameraCapture.read()
        while success and cv2.waitKey == -1:
            success, frame = cameraCapture.read()
            gray = cv2.cvtco1or(frame, cv2.coLoR_BGR2GRAY)
            faces = face_cascade.detectMultiScal
            for (x, y, w, h) in faces:
                ROI = gray[x:x + w, y:y + h]
                ROI = cv2.resize(ROI, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                label, prob = self.model.predict(ROI)
                if prob > threshold:
                    show_name = name_list[label]
                else:
                    show_name = "Stranger"
                cv2.putТext(frame, show_name, (x, y - 20).ev2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.imshow("Camera", frame)
            else:
                cameraCapture.release()
                cv2.destroyALLWindows()


if __name__ == '__main__':
    camera = Camera_reader()
    camera.build_camera()
