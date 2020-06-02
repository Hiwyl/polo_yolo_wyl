"""自动划分训练集、测试集"""

import os
import random

trainval_percent = 0.1
train_percent = 0.9
xmlfilepath = 'xml'
txtsavepath = './txts'
if not os.path.exists(txtsavepath):
    os.mkdir(txtsavepath)
total_xml = os.listdir(xmlfilepath)


num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftest = open('txts/test.txt', 'w')
ftrain = open('txts/train.txt', 'w')
fval = open('txts/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftest.write(name)
        fval.write(name)
    else:
        ftrain.write(name)

ftrain.close()
fval.close()
ftest.close()
