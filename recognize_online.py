import pandas as pd
import datetime
import requests
from io import BytesIO
import time
import json
from config import WORK_SETTING
def test_dir():
    img_data = []
    root = WORK_SETTING['root']
    datacsv = 'testDataInfo.csv'
    with open(f'{root}/{datacsv}', 'r') as f:
        for lines in f.readlines():
            line = lines.rstrip().split(',')
            if line[0] != '':
                path = line[0]
                label = line[1]
                test_data = {}
                with open(path, 'rb') as f:
                    content = f.read()
                s = time.time()
                url = "http://127.0.0.1:6000/b"
                image_file_name = 'captcha.{}'.format(1)
                files = {'image_file': (image_file_name, BytesIO(content), 'application')}
                r = requests.post(url=url, files=files)
                e = time.time()
                print("识别用时{}".format(e - s))
                target = json.loads(r.text)["value"]
                now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                test_data['图片路径'] = path
                test_data['真实值'] = label
                test_data['预测值'] = target
                test_data['预测正确'] = (int(label) == int(target))
                print(test_data)
                img_data.append(test_data)
    pd.DataFrame(img_data).to_excel("{}.xlsx".format('output'), engine='xlsxwriter')

def API(path):
    with open(path, 'rb') as f:
        content = f.read()
    s = time.time()
    url = "http://127.0.0.1:6000/b"
    image_file_name = 'captcha.{}'.format(1)
    files = {'image_file': (image_file_name, BytesIO(content), 'application')}
    r = requests.post(url=url, files=files)
    e = time.time()
    print("识别用时{}ms".format((e - s)*1000))
    target = json.loads(r.text)["value"]
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(target)
if __name__ == '__main__':

    path =r'D:\Py\基于pytorch的像素分类\images\trainData\0.jpg'
    API(path)


