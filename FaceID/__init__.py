from .getCameraPics import cameraAutoForPictures
from .dataHelper import readPicSaveFace
from .faceRegnigtionModel import Model, DataSet
from .cameraDemo import Camera_reader

dataDir = '../data/'
datasetDir = '../dataset/'

if __name__ == '__main__':
    username = "ghm"
    cameraAutoForPictures(username)
    dataDir = f'../data/{username}/'
    datasetDir = f'../dataset/{username}/'
    print(f'源文件夹：{dataDir}\n目的文件夹：{datasetDir}')
    readPicSaveFace(dataDir, datasetDir, '.jpg', '.JPG', 'png', 'PNG', 'tiff')
    camera = Camera_reader()
    camera.build_camera()
