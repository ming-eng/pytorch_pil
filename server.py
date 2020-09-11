from config import  WORK_SETTING
from torchvision.transforms import  transforms

import json
from io import BytesIO
import os
import time
from flask import Flask, request, jsonify, Response
from PIL import Image
from Net.model import Net2
import torch
# Flask对象
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))

#加载识别路径
model_path =WORK_SETTING['model_path']


def response_headers(content):
    resp = Response(content)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


@app.route('/b', methods=['POST'])
def up_image():
    if request.method == 'POST' and request.files.get('image_file'):
        net = Net2()
        # if torch.cuda.is_available():
        #     net = net.cuda()
        net.eval()
        net.load_state_dict(torch.load(model_path))
        ToTnesor = transforms.ToTensor()
        timec = str(time.time()).replace(".", "")
        #获取突破
        file = request.files.get('image_file')
        #读取突破
        img = file.read()
        #转入内存
        img = BytesIO(img)

        #打开图片
        img=Image.open(img).convert('1')
        img.save('test.png')

        #转化图片
        img = ToTnesor(img)
        #图片增维
        img.unsqueeze_(0)
        # img = img.cuda()
        output = net(img)
        s = time.time()
        value = output.argmax(dim=1).item()
        e = time.time()
        result = {
            'time': timec,   # 时间戳
            'value': value,  # 预测的结果
            'speed_time(ms)': int((e - s) * 1000)  # 识别耗费的时间
        }
        return jsonify(result)
    else:
        content = json.dumps({"error_code": "1001"})
        resp = response_headers(content)
        return resp


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=6000,
        debug=True
    )
