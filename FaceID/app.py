from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask import redirect, url_for, render_template
import os
import cv2
import time
import numpy as np
from flask import Flask
from flask import request

from FaceID.file import read_name_list
from faceRegnigtionModel import Model
from cameraDemo import Camera_reader

app = Flask(__name__)
app.config['UPLOADED_PHOTO_DEST'] = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOADED_PHOTO_ALLOW'] = IMAGES


def dest(name):
    return f'{app.config.UPLOADED_PHOTO_DEST}/{name}'


photos = UploadSet('PHOTO')
configure_uploads(app, photos)


@app.route('/photo/<name>')
def show(name):
    if name is None:
        print('出错了!')
    url = photos.url(name)

    def detectOnePicture(path):
        model = Model()
        model.load()
        img = cv2.imread(path)
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        picType, prob = model.predict(img)
        if picType != -1:
            name_list = read_name_list('../dataset/')
            print(name_list[picType], prob)
            res = u"识别为：" + name_list[picType] + u"的概率为：" + str(prob)
        else:
            res = u"抱歉，未识别出该人！请尝试增加数据量来训练模型"
        return res

    if request.method == "GET":
        picture = name
    start_time = time.time()
    res = detectOnePicture(picture)
    end_time = time.time()
    execute_time = str(round(end_time - start_time, 2))
    tsg = u'总耗时为：%s 秒' % execute_time
    return render_template('show.html', url=url, name=name, xinxi=res, shijian=tsg)


@app.route("/")
def init():
    return render_template("index.html", title='Home')


@app.route("/she/")
def she():
    camera = Camera_reader()
    camera.build_camera()
    return render_template("index.html", title="Home")


if __name__ == '__main__':
    print('faceRegnitionDemo')
    app.run(debug=True, host="0.0.0.0")
