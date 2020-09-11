import torch
from PIL import Image
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, root, datacsv, transform=None):
        super(MyDataset, self).__init__()
        print(f'{root}/{datacsv}')
        with open(f'{root}/{datacsv}', 'r') as f:
            imgs = []
            # 读取csv信息到imgs列表
            for lines in f.readlines():
                line= lines.rstrip().split(',')
                if line[0]!='':
                    path=line[0]
                    label=line[1]
                    imgs.append((path, int(label)))
        self.imgs = imgs
        self.transform = transform if transform is not None else lambda x: x

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = self.transform(Image.open(path).convert('1'))
        return img, label

    def __len__(self):
        return len(self.imgs)