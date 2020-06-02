# -*- coding: utf-8 -*-
"""
@Author :       wyl
@Email :  wangyl306@163.com
@Date :      2020/6/1
"""

import cv2
import numpy as np
import os
import time
import poly_yolo as yolo

#load pretrained model
#if you want to detect more objects, lower the score and vice versa
trained_model = yolo.YOLO(model_path='./weights/ep001-loss60.854-val_loss21.772.h5', iou=0.5, score=0.3)


#helper function
def translate_color(cls):
    if cls == 0: return (230, 25, 75)
    if cls == 1: return (60, 180, 75)
    if cls == 2: return (255, 225, 25)
    if cls == 3: return (0, 130, 200)
    if cls == 4: return (245, 130, 48)
    if cls == 5: return (145, 30, 180)
    if cls == 7: return (70, 240, 240)
    if cls == 8: return (240, 50, 230)
    if cls == 9: return (210, 245, 60)
    if cls == 10: return (250, 190, 190)
    if cls == 11: return (0, 128, 128)
    if cls == 12: return (230, 190, 255)
    if cls == 13: return (170, 110, 40)
    if cls == 14: return (255, 250, 200)
    if cls == 15: return (128, 0, 128)
    if cls == 16: return (170, 255, 195)
    if cls == 17: return (128, 128, 0)
    if cls == 18: return (255, 215, 180)
    if cls == 19: return (80, 80, 128)

dir_imgs_name = './dataset/imgs' #path_where_are_images_to_clasification
out_path = 'poly_yolo_predict/' #path, where the images will be saved. The path must exist
if not os.path.exists(out_path):
    os.mkdir(out_path)
list_of_imgs = [root+"/"+name for root, dirs, files in os.walk(dir_imgs_name) for name in files]
list_of_imgs.sort()

# browse all images
total_boxes = 0
imgs = 0
for im in range(0, len(list_of_imgs)):
    imgs += 1
    img = cv2.imread(list_of_imgs[im])
    overlay = img.copy()
    boxes = []
    scores = []
    classes = []

    # realize prediciction using poly-yolo
    startx = time.time()
    box, scores, classes, polygons = trained_model.detect_image(img)
    print('Prediction speed: ', 1.0 / (time.time() - startx), 'fps')

    # example, hw to reshape reshape y1,x1,y2,x2 into x1,y1,x2,y2
    for k in range(0, len(box)):
        boxes.append((box[k][1], box[k][0], box[k][3], box[k][2]))
        cv2.rectangle(img, (box[k][1], box[k][0]), (box[k][3], box[k][2]), translate_color(classes[k]), 3, 1)
    total_boxes += len(boxes)

    # browse all boxes
    for b in range(0, len(boxes)):
        f = translate_color(classes[b])
        points_to_draw = []
        offset = len(polygons[b]) // 3

        # filter bounding polygon vertices
        for dst in range(0, len(polygons[b]) // 3):
            if polygons[b][dst + offset * 2] > 0.3:
                points_to_draw.append([int(polygons[b][dst]), int(polygons[b][dst + offset])])

        points_to_draw = np.asarray(points_to_draw)
        points_to_draw = points_to_draw.astype(np.int32)
        if points_to_draw.shape[0] > 0:
            cv2.polylines(img, [points_to_draw], True, f, thickness=2)
            cv2.fillPoly(overlay, [points_to_draw], f)

    img = cv2.addWeighted(overlay, 0.4, img, 1 - 0.4, 0)
    cv2.imwrite(out_path + str(imgs) + '.jpg', img)

print('total boxes: ', total_boxes)
print('imgs: ', imgs)