import math
import os

import numpy as np
import speech_recognition as sr
import cv2
from PIL import Image
from pydub import AudioSegment

from flask import Flask, render_template, request, redirect, send_file, app

UPLOAD_FOLDER = os.path.join('static', 'uploads')


application = Flask(__name__)
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@application.route("/", methods=["GET", "POST"])
def index():
    transcript = ""
    rus_mode = request.form.get('rus-lang', 'off')
    if request.method == "POST":
        print("FORM DATA RECEIVED")
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            return redirect(request.url)
        if file:
            recognizer = sr.Recognizer()
            x = AudioSegment.from_file(file.stream)
            file = x.export(format="wav")
            audio_file = sr.AudioFile(file)
            with audio_file as source:
                data = recognizer.record(source)
            if rus_mode == 'on':
                transcript = recognizer.recognize_google(data, language="ru-Ru")
            else:
                transcript = recognizer.recognize_google(data)

    return render_template('index.html', transcript=transcript)

@application.route("/img", methods=["GET", "POST"])
def indexImg():
    if request.method == "POST":
        print("FORM DATA RECEIVED")
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        path = os.path.join(application.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)
        proccesed = image_preprocess(os.path.join(application.config['UPLOAD_FOLDER'], file.filename))
        result = os.path.join(application.config['UPLOAD_FOLDER'], "result.jpg")


    return render_template('index.html', cropResult=result)


@application.route('/recognition/audio/mp3', methods=['POST'])
def recognise_mp3():
    if request.method == 'POST':
        f = request.files['file']

    x = AudioSegment.from_file(f.stream)
    file = x.export(format="wav")

    mp3 = sr.AudioFile(file)

    r = sr.Recognizer()

    with mp3 as source:
        audio = r.record(source)
    rec = r.recognize_google(audio)
    return rec


@application.route('/recognition/sts/jpg/crop', methods=['POST'])
def crop_sts():
    if request.method == 'POST':
        f = request.files['file']

    path = os.path.join(application.config['UPLOAD_FOLDER'], f.filename)
    f.save(path)

    proccesed = image_preprocess(os.path.join(application.config['UPLOAD_FOLDER'], f.filename))
    return send_file(os.path.join(application.config['UPLOAD_FOLDER'], "result.jpg"), mimetype='image/gif')
    # return send_file(proccesed, mimetype='image/gif')


def image_preprocess(raw_image):
    print(raw_image)
    img = cv2.imread(raw_image)
    temp = cv2.fastNlMeansDenoising(img, h=7)

    # ksize
    ksize = (30, 30)

    # Using cv2.blur() method
    temp = cv2.blur(temp, ksize, cv2.BORDER_DEFAULT)
    cv2.imwrite('output/blur.jpg', temp)

    image = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    T_, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite('output/thresholded2.jpg', thresholded)

    edges = cv2.Canny(thresholded, 0, 10)
    cv2.imwrite('output/edges2.jpg', edges)

    # imagem = cv2.cvtColor(thresholded)
    imagem = ~thresholded
    cv2.imwrite('output/reversed.jpg', imagem)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

    # Applying dilation on the threshold image
    dilation = cv2.dilate(imagem, rect_kernel, iterations=1)
    cv2.imwrite('output/dilation.jpg', dilation)

    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnt = contours[0]

    image_copy = img.copy()

    # compute the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(cnt)

    # draw contour
    img_with_counters = cv2.drawContours(image_copy, [cnt], 0, (0, 255, 255), 2)
    cv2.imwrite('output/image_counters1.jpg', img_with_counters)

    cropped_images = []
    max = 0
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])
        print(area)
        if (area > max):
            max = area
            x, y, w, h = cv2.boundingRect(contours[i])
            cropped_img = img[y:y + h, x:x + w]
            cropped_images.append(cropped_img)
            img_name = os.path.join(application.config['UPLOAD_FOLDER'], "result.jpg")
            cv2.imwrite(img_name, cropped_img)

    # draw the bounding rectangle
    #final_image = cv2.rectangle(img_with_counters, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #cv2.imwrite('output/image_counters2.jpg', final_image)

    return cropped_images[0]


if __name__ == "__main__":
    application.run()
