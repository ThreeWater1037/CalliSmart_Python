
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


#从Http获取图片
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


#下载图片并保存
img = fetchImageFromHttp(
    'https://aminos-callismart.oss-cn-beijing.aliyuncs.com/myai.jpg')

#将图片储存在本地
plt.imsave('img.jpg', img)

file_path = "img.jpg"
#识别图片中的文字
text = baidu_identify.character_identification(file_path)
#返回识别结果
print(text)

#查找图片
image_path = new_retrieve.find_image("numbers.txt", text, "good")
img1 = cv2.imread(file_path, 0)
img2 = cv2.imread(image_path, 0)

iou_score, similarity_score, pearson_score, adjust = structure_evaluation.preprocessing(img1, img2)
print(iou_score, similarity_score, pearson_score)#返回多元相似度
