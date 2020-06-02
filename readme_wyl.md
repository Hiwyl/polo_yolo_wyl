**环境配置：**

+   Python 3.5.x
+   Keras 2.3.x
+   Tensorflow 1.6.x
+   Numpy，PIL，OpenCV

**数据集：**

1.  labelimg标注：imgs/xml两个文件夹

2.  划分数据集-*注意修改自己的类别名*

    ```python
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
    ```

3.  生成训练所需txt

    ```python
    # -*- coding: utf-8 -*-
    """
    @Author :       wyl
    @Email :  wangyl306@163.com
    @Date :      2020/5/31
    """
    import xml.etree.ElementTree as ET
    
    sets=[('wyl', 'train'), ('wyl', 'val'), ('wyl', 'test')]
    
    classes = ["obj","obj1"]
    
    def convert_annotation(year, image_id, list_file):
        in_file = open('xml/%s.xml'% image_id, 'r', encoding='utf-8')
        tree=ET.parse(in_file)
        root = tree.getroot()
    
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult)==1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
            list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
    
    
    for year, image_set in sets:
        image_ids = open('txts/%s.txt'%image_set).read().strip().split()
        list_file = open('%s_%s.txt'%(year, image_set), 'w')
        for image_id in image_ids:
            list_file.write('%s/imgs/%s.jpg'%("./dataset", image_id))
            convert_annotation(year, image_id, list_file)
            list_file.write('\n')
        list_file.close()
    ```

**训练：**

1.  修改models/yolo_classes.txt为自己的类别
2.  修改polo_yolo.py中参数
    +   phase：是否加载预训练权重（1-不加载；2-加载）
    +   Line：803-815

 python polo_yolo.py

**预测**：

  python pred.py