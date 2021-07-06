from context import FaceID
import cv2
import os


def test_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    width, height, w = 640, 480, 360
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ret, frame = cap.read()
    if not ret:
        return False
    return True


def test_getCameraPics():
    print("测试文件getCameraPics.py")
    if not test_camera():
        print("无法打开摄像头")
    else:
        print("开始拍照，p for picture, q for quit, c for new dir")
        print("照片存储目录为", saveDir)
        FaceID.cameraAutoForPictures(saveDir)


def test_dataHelper():
    print("测试文件dataHelper.py")
    print("开始原始图像预处理......")
    print("从{}获取图像，保存结果存放在{}".format(saveDir, datasetDir))
    FaceID.readPicSaveFace(saveDir, datasetDir, '.jpg', '.JPG', 'png', 'PNG', 'tiff')


def test_faceRegnitionModel():
    print("测试文件faceRegnitionModel.py")
    print("训练模型")
    dataset = FaceID.DataSet('../dataset/')
    model = FaceID.Model()
    model.read_trainData(dataset)
    model.build_model()
    model.train_model()
    model.evaluate_model()
    model.save()


def test_cameraDemo():
    print("测试文件cameraDemo.py")
    camera = FaceID.Camera_reader()
    camera.build_camera()


if __name__ == '__main__':
    username = "gonghaoming"
    saveDir = "../data/{}/".format(username)
    datasetDir = "../dataset/{}/".format(username)

    # test_getCameraPics()
    # print('\n\n\n')
    # test_dataHelper()
    # print('\n\n\n')
    # test_faceRegnitionModel()
    # print('\n\n\n')
    test_cameraDemo()
