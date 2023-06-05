import math
import os

import numpy as np
import speech_recognition as sr
import cv2
from easyocr import Reader


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
    return send_file(os.path.join(application.config['UPLOAD_FOLDER'], f.filename + "_result.jpg"), mimetype='image/gif')
    # return send_file(proccesed, mimetype='image/gif')

@application.route('/recognition/sts', methods=['POST'])
def rec_sts():
    if request.method == 'POST':
        f = request.files['file']

    path = os.path.join(application.config['UPLOAD_FOLDER'], f.filename)
    f.save(path)

    proccesed = image_preprocess(os.path.join(application.config['UPLOAD_FOLDER'], f.filename))
    return recognise_sts(proccesed)


def image_preprocess(raw_image):
    img = cv2.imread(raw_image)
    temp = cv2.fastNlMeansDenoising(img, h=7)

    # ksize
    ksize = (30, 30)

    # Using cv2.blur() method
    temp = cv2.blur(temp, ksize, cv2.BORDER_DEFAULT)
    cv2.imwrite('output/blur.jpg', temp)

    image = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('output/gray.jpg', image)

    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel1)
    div = np.float32(image) / (close)
    image = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))
    cv2.imwrite('output/normalize.jpg', image)

    T_, thresholded = cv2.threshold(image,0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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

    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                     lineType=cv2.LINE_AA)
    # compute the bounding rectangle of the contour
    cv2.imwrite('output/image_counters.jpg', image_copy)

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
            img_name = raw_image + "_result.jpg"
            cv2.imwrite(img_name, cropped_img)

    # draw the bounding rectangle
    #final_image = cv2.rectangle(img_with_counters, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #cv2.imwrite('output/image_counters2.jpg', final_image)

    ##обработка для распзнавания

  #  recognise_sts(cropped_images[1].copy())

    # kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, 0]], np.float32)
   # kernel = 1 / 3 * kernel
   # dst = cv2.filter2D(image_for_proccess, -1, kernel)
   # cv2.imwrite('output/cropped/sharp.jpg', dst)


    #edges = cv2.Canny(dst, 100, 150)
    #cv2.imwrite('output/cropped/edges_cropped.jpg', edges)

    return cropped_images[-1]

def recognise_sts(image_for_proccess):
    # get dimensions of image
    dimensions = image_for_proccess.shape

    # height, width, number of channels in image
    height = image_for_proccess.shape[0]
    width = image_for_proccess.shape[1]
    channels = image_for_proccess.shape[2]

    print('Image Dimension    : ', dimensions)
    print('Image Height       : ', height)
    print('Image Width        : ', width)
    print('Number of Channels : ', channels)

    image_for_proccess = image_for_proccess[height//10:round(height//2.5), width//12:round(width//1.2)].copy()

    # image[yMin:yMax, xMin:xMax]

    cv2.imwrite('output/cropped/cropped.jpg', image_for_proccess)

    image_for_proccess = cv2.resize(image_for_proccess, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    image_for_proccess = cv2.cvtColor(image_for_proccess, cv2.COLOR_BGR2GRAY)

    cv2.imwrite('output/cropped/gray.jpg', image_for_proccess)

    kernel = np.ones((1, 1), np.uint8)
    image_for_proccess = cv2.dilate(image_for_proccess, kernel, iterations=1)
    image_for_proccess = cv2.erode(image_for_proccess, kernel, iterations=1)

    cv2.imwrite('output/cropped/noise.jpg', image_for_proccess)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
    image_for_proccess = clahe.apply(image_for_proccess)

    #  blur = cv2.GaussianBlur(image_for_proccess, (5, 5), 0)
    #  ret3, image_for_proccess = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # T_, image_for_proccess = cv2.threshold(image_for_proccess, 119, 255, cv2.THRESH_BINARY)

    image_for_proccess = cv2.GaussianBlur(src=image_for_proccess, ksize=(3, 3), sigmaX=0, sigmaY=0)
    #  image_for_proccess = cv2.bilateralFilter(image_for_proccess, 9, 75, 75)
    _, image_for_proccess = cv2.threshold(image_for_proccess, thresh=120, maxval=255,
                                          type=cv2.THRESH_TRUNC + cv2.THRESH_OTSU)

    # image_for_proccess = cv2.adaptiveThreshold(cv2.bilateralFilter(image_for_proccess, 9, 75, 75),
    # 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,31, 11)

    # image_for_proccess = cv2.adaptiveThreshold(cv2.medianBlur(image_for_proccess, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 27, 8)
    # image_for_proccess = cv2.adaptiveThreshold(cv2.GaussianBlur(image_for_proccess, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 27, 8)
    for_rec = image_for_proccess.copy()

    cv2.imwrite('output/cropped/thres.jpg', image_for_proccess)

    # contours_new, _ = cv2.findContours(~image_for_proccess, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # img_with_counters = cv2.drawContours(cropped_images[-1], [contours_new[0]], 0, (0, 255, 255), 2)

    # cv2.imwrite('output/cropped/counters.jpg', img_with_counters)
    # for_rec = cv2.dilate(for_rec, kernel, iterations=5)

    #for_rec = cv2.resize(for_rec, None, fx=2, fy=2  , interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('output/cropped/for_recognition.jpg', for_rec)

    # text = pytesseract.image_to_string(for_rec, lang='eng+rus')
    # print(text)

    reader = Reader(lang_list={'en', 'ru'})
    results = reader.readtext(for_rec)
    rec_result = []
    line = 0
    prev = ""
    vin = ""
    vin_prob = 0
    for (bbox, text, prob) in results:
        # display the OCR'd text and associated probability
        rec_result.append(dict(line=line, prob=prob, text=text))
        line = line + 1
        if ('VIN' in prev.upper()):
            vin = text
            vin_prob = prob
        print("[INFO] {:.4f}: {}".format(prob, text))
        prev = text
    rec_result.append(dict(line="vin", prob=vin_prob, text=vin))
    return rec_result



if __name__ == "__main__":
    application.run()
