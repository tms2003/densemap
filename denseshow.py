# -*- coding:utf-8 -*-
#!/usr/bin/env python

import argparse
import json
import matplotlib.pyplot as plt

from labelme import utils
import numpy as np
import scipy.io as sio
import cv2
import os


def dense(img, oldpoints):

    points=[]
    for item in oldpoints:
        for point in item["points"]:
            points.append(point);

    path = 'dense/IMG_'
    ipath = 'dense/IMG_'
    
    R = 50
    r = np.sqrt((R // 2) ** 2 * 2)

    mp = sio.loadmat('dense/map.mat')
    mp = mp['c']
    mp = mp[::-1]
    out_path = './density_map/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    


    n = len(points)  # 一个有多少个标记点
    P = points   #  [[100, 100], [200, 200]] #点的数组

    # 读取原始图像
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    N = img.shape[0]
    M = img.shape[1]
    den_map = np.zeros([N, M, 3], dtype=np.float)
    tot = np.zeros([N, M], dtype=np.float)
    for j in range(n):
        y = int(P[j][0])  # 取标记点的y坐标
        x = int(P[j][1])  # 取标记点的x坐标
        # 处理周围100以内的图像点
        for X in range(x - R // 2, x + R // 2 + 1):
            if X < 0 or X >= N:
                continue
            for Y in range(y - R // 2, y + R // 2 + 1):
                if Y < 0 or Y >= M:
                    continue
                dis = np.sqrt((X - x) ** 2 + (Y - y) ** 2)  # 求距离
                add = np.exp(dis * (-4) / r)  # 得出数据
                #            if j == 1:
                #                print(add)
                tot[X][Y] += add
    max_den = tot.max()
    for X in range(N):
        for Y in range(M):
            pixel = 255 * tot[X][Y] / max_den
            den_map[X][Y] = mp[int(pixel)] * 255
            den_map[X][Y] = [int(ele) for ele in den_map[X][Y]]
    cv2.imwrite(out_path + 'density_map_out.jpg', den_map)
    print('\\' + str(len))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    args = parser.parse_args()

    json_file = args.json_file

    data = json.load(open(json_file)) # 加载json文件

    img = utils.img_b64_to_arr(data['imageData']) # 解析原图片数据

    dense(img, data['shapes']);


    #src=cv2.imread('json/bird.jpg')   

    #cv2.imshow('input_image',img)

    #lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes']) # 解析'shapes'中的字段信息，解析出每个对象的mask与对应的label   lbl存储 mask，lbl_names 存储对应的label
    # lal 像素取值 0、1、2 其中0对应背景，1对应第一个对象，2对应第二个对象
    # 使用该方法取出每个对象的mask mask=[] mask.append((lbl==1).astype(np.uint8)) # 解析出像素值为1的对象，对应第一个对象 mask 为0、1组成的（0为背景，1为对象）
    # lbl_names  ['background','cat_1','cat_2']

    #captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
    #lbl_viz = utils.draw_label(lbl, img, captions)
    a = 1 
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()


