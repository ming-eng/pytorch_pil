from Net.model import Net2
from config import  WORK_SETTING
import torch
from PIL import  Image
from torchvision.transforms import Compose, ToTensor
from torchvision.transforms import  transforms
import torch.utils.data as Data
from dataset.loaddata import MyDataset
import pandas as pd
model_path =WORK_SETTING['model_path']

def predict_local_one(path):
    net=Net2()
    if torch.cuda.is_available():
        net = net.cuda()
    net.eval()
    net.load_state_dict(torch.load(model_path))
    ToTnesor=transforms.ToTensor()
    img =ToTnesor(Image.open(path).convert('1'))
    img.unsqueeze_(0)
    img=img.cuda()
    output=net(img)
    lable=output.argmax(dim=1).item()
    return lable

def predict_local_dir():
    img_data=[]
    root=WORK_SETTING['root']
    datacsv='testDataInfo.csv'
    with open(f'{root}/{datacsv}', 'r') as f:
        # 读取csv信息到imgs列表
        for lines in f.readlines():
            line = lines.rstrip().split(',')
            if line[0] != '':
                path = line[0]
                label = line[1]
                test_data={}
                target=predict_local_one(path)
                test_data['图片路径']=path
                test_data['真实值']=label
                test_data['预测值']=target
                test_data['预测正确']=(int(label)==int(target))
                img_data.append(test_data)
    pd.DataFrame(img_data).to_excel("{}.xlsx".format('output'), engine='xlsxwriter')


if __name__ == '__main__':
    path='./images/trainData/51.jpg'
    predict_local_dir()
