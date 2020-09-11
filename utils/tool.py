import collections
import itertools


def chooseData(dataset, scale):
    # 将类别为1的排序到前面
    dataset.imgs.sort(key=lambda x: x[1], reverse=True)
    # 获取类别1的数目 ，取scale倍的数组，得数据不那么偏斜
    trueNum = collections.Counter(itertools.chain.from_iterable(dataset.imgs))[1]
    end = min(trueNum * scale, len(dataset))
    dataset.imgs = dataset.imgs[:end]