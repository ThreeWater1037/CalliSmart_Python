import sys
import json
import base64
import requests
import ssl
from urllib.request import urlopen
from urllib.request import Request
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.parse import quote_plus

API_KEY = 'hqQY4N9MMvE6g93jcyMD262f'
SECRET_KEY = 'yIPLjPnqCToYwh9blNpddvk1dqa9OZs0'
file_path = './oyx/0001.jpg'


def fetch_token():
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id='+API_KEY+'&client_secret='+SECRET_KEY
    response = requests.get(host)
    if response:
        result = response.json()
        return result['access_token']


def read_file(image_path):
    f = open(image_path, 'rb')
    return f.read()


def ocr(token, picture_file):
    img = base64.b64encode(picture_file)
    params = {"image":img,"recognize_granularity":"big","detect_direction":"true"}
    access_token = token
    request_url = 'https://aip.baidubce.com/rest/2.0/ocr/v1/handwriting'
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}

    response = requests.post(request_url, data=params, headers=headers,)
    if response:
        return response.json()

def character_identification(file_path):
    token = fetch_token()
    picture_file = read_file(file_path)
    result_json = ocr(token, picture_file)

    text = ""
    for words_result in result_json["words_result"]:
        text = text + words_result["words"]
    return text




'''if __name__ == '__main__':

    token = fetch_token()
    picture_file = read_file(file_path)
    result_json = ocr(token, picture_file)

    text = ""
    for words_result in result_json["words_result"]:
        text = text + words_result["words"]
    print(text)
    print(result_json["words_result"])'''