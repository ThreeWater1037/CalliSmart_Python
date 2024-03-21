import base64
import urllib
import sys
import os
from matplotlib import pyplot as plt
import structure_evaluation
import new_retrieve
import numpy as np
import cv2
import baidu_identify
import cv2
import numpy as np
import urllib.request
from flask import Flask, request, jsonify, send_file
import os

app = Flask(__name__)


@app.route('/evaluate', methods=['POST'])
def evaluate_image():
    try:
        print(1)
        print(request.headers)
        print(request.form)
        # 从请求中获取图片 URL 和字体参数
        image_url = request.form['exurl']
        font = int(request.form['font'])

        print(image_url)
        print(font)
        # 在这里进行评测算法的处理
        # 这里只是一个示例，可以根据实际情况进行修改
        # word, score, evaluation_image_path = evaluate(image_url, font)
        word, score, url = evaluate(image_url, font)
        print(word, score, url)

        # 返回评测结果
        # return jsonify({"word": word, "score": score}), send_file(evaluation_image_path, mimetype='image/jpeg'), 200
        return jsonify({"score": score, "word": word, "url": url}), 200
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Internal server error"}), 500


def image_to_bytes(image_path):
    with open(image_path, 'rb') as image_file:
        image_bytes = image_file.read()
        image_str = base64.b64encode(image_bytes).decode('utf-8')
    return image_str

def fetchImageFromHttp(image_url, timeout_s=1):
    try:
        if image_url:
            resp = urllib.request.urlopen(image_url, timeout=timeout_s)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            return image
        else:
            return []
    except Exception as error:
        print('获取图片失败', error)
        return []


def evaluate(image_url, font):
    # 在这里编写评测算法的代码
    # 可以从图片 URL 中下载图片数据，然后进行评测
    # 这里只是一个示例，可以根据实际情况进行修改

    img = fetchImageFromHttp(
        image_url)
    plt.imsave('img.jpg', img)
    file_path = "img.jpg"
    text = baidu_identify.character_identification(file_path)

    print(text)

    image_path = new_retrieve.find_image("numbers.txt", text, "good")
    img1 = cv2.imread(file_path, 0)
    img2 = cv2.imread(image_path, 0)

    iou_score, similarity_score, pearson_score, adjust = structure_evaluation.preprocessing(img1, img2)
    print(iou_score, similarity_score, pearson_score)




    word = text
    score = (iou_score+similarity_score+pearson_score)/3
    url = image_to_bytes("adjust.jpg")

    # return word, score, evaluation_image_path
    return word, score, url


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
