import csv
import collections
import os
import shutil
import numpy as np
from PIL import Image

def buildDataset(root, dataType, dataSize):
    """构造数据集
    构造的图片存到root/{dataType}Data
    图片地址和标签的csv文件存到 root/{dataType}DataInfo.csv
    Args:
      root:str
        项目目录
      dataType:str
        'train'或者‘test'
      dataNum:int
        数据大小
    Returns:
    """
    dataInfo = []
    dataPath = f'{root}/{dataType}Data'
    if not os.path.exists(dataPath):
        os.makedirs(dataPath)
    else:
        shutil.rmtree(dataPath)
        os.mkdir(dataPath)

    for i in range(dataSize):
        # 创建0，1 数组
        imageArray = np.random.randint(0, 2, (4, 4))
        # 计算0，1数量得到标签
        allBlackNum = collections.Counter(imageArray.flatten())[0]
        innerBlackNum = collections.Counter(imageArray[1:3, 1:3].flatten())[0]
        label = 0 if (allBlackNum - innerBlackNum) > innerBlackNum else 1
        # 将图片保存
        path = f'{dataPath}/{i}.jpg'
        dataInfo.append([path, label])
        im = Image.fromarray(np.uint8(imageArray * 255))
        im = im.convert('1')
        im.save(path)
    # 将图片地址和标签存入csv文件
    filePath = f'{root}/{dataType}DataInfo.csv'
    print(dataInfo)
    with open(filePath, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(dataInfo)


if __name__ == '__main__':
    buildDataset('./images','test',18)