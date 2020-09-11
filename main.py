from dataset.loaddata import MyDataset
from config import WORK_SETTING
from torchvision.transforms import  transforms
from Net.model import Net2
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch
import os

import torch.utils.data as Data
net2 = Net2()


#写入日志
writer = SummaryWriter()
#保存模型
model_path =WORK_SETTING['model_path']
lr=WORK_SETTING['lr']
restor =WORK_SETTING['restore']
lr=WORK_SETTING['lr']
numEpochs=WORK_SETTING['numEpochs']
batchSize=WORK_SETTING['batchSize']
root=WORK_SETTING['root']


#判断模型文件是否存在
if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')

#判断cuda是否可以使用
if torch.cuda.is_available():
    net2 .cuda()
    print("cuda is available, and the calculation will be moved to GPU\n")
else:
    print("cuda is unavailable!")


if restor:
    net2.load_state_dict(torch.load(model_path))


trainData=MyDataset(root = root,datacsv='trainDataInfo.csv', transform=transforms.ToTensor())
testData=MyDataset(root = root,datacsv='testDataInfo.csv', transform=transforms.ToTensor())
trainIter = Data.DataLoader(dataset=trainData, batch_size=batchSize, shuffle=True)
testIter = Data.DataLoader(dataset=testData, batch_size=batchSize)




# 交叉熵损失函数
loss = nn.CrossEntropyLoss()

optimizer2 = torch.optim.SGD(net2.parameters(),lr = lr)

def evaluateAccuracy(dataIter, net):
  accSum, n = 0.0, 0
  with torch.no_grad():
    for X, y in dataIter:
      if torch.cuda.is_available():
        X = X.cuda()
        y = y.cuda()
      accSum += (net(X).argmax(dim=1) == y).float().sum().item()
      n += y.shape[0]
  return accSum / n

def train(net, trainIter, testIter, loss, numEpochs, batchSize,
       optimizer):
  for epoch in range(numEpochs):
    trainLossSum, trainAccSum, n = 0.0, 0.0, 0
    for X,y in trainIter:
      if torch.cuda.is_available():
        X =X.cuda()
        y= y.cuda()
      yHat = net(X)
      l = loss(yHat,y).sum()
      optimizer.zero_grad()
      l.backward()
      optimizer.step()
      # 计算训练准确度和loss
      trainLossSum += l.item()
      trainAccSum += (yHat.argmax(dim=1) == y).sum().item()
      n += y.shape[0]
    # 评估测试准确度
    testAcc = evaluateAccuracy(testIter, net)
    torch.save(net.state_dict(), model_path)
    print('epoch {:d}, loss {:.4f}, train acc {:.3f}, test acc {:.3f}'.format(epoch + 1, trainLossSum / n, trainAccSum / n, testAcc))
train(net2, trainIter, testIter, loss, numEpochs, batchSize,optimizer2)