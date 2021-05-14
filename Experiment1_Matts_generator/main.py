# -*- encoding: utf-8 -*-
'''
@Time        :2021/03/17 19:34:44
@Author      :Qu Yue
@File        :main.py
@Software    :Visual Studio Code
Introduction:  
'''
#%%
# %matplotlib qt5
import numpy as np
import pandas as pd
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as imgplt
import random
import time
import json
import os
from tqdm import tqdm
from blocks import Text_Selector, MattsPage
from draw_tools import image_overlay, make_pic_lights
from image_enhance import Image_Enhance#%%

class Parameters():
    def __init__(self):
        ##### save
        self.save_path = '../../Datasets/Dataset0/'
        self.save_type = 'train'
        ##### blocks
        self._top_edge = [0, 0.15] # 上边缘比例
        self._bottom_edge = [0, 0.15] # 下边缘比例
        self._left_edge = [0, 0.2] # 左边缘比例
        self._right_edge = [0, 0.2] # 右边缘比例

        self._top_edge_line = [3, 10] # 上边缘线宽度
        self._bottom_edge_line = [3, 10] # 下边缘线宽度
        self._left_right_edge_line = [2, 7] # 左和右边缘线宽度
        self._edge_distance = [0, 10] # 边缘线和格之间的距离

        self._block_size = [50, 100] # 田字格的像素大小

        self._block_row_num = [8, 15] # 有几行
        self._block_column_num = [5, 10] # 有几列
        self._row_type = ['nodistance', 'distance', 'threelines'] # 行间隔类型
        self._column_type = ['nodistance', 'distance'] # 列间隔类型
        self._row_line = ['no', 'edge', 'each'] # 行线
        self._column_line = ['no', 'edge','each'] # 列线

        self._block_type = ['matts', 'matts', 'grid', 'block'] # 三种田字格（田字、米字、方块）
        self._block_line_type = ['dotted0','dotted1', 'dotted2'] # 三种虚线
        self._block_line_gap = [6, 10] # 虚线的间距（分之一）

        self._threelines_type = ['dotted0','dotted1', 'normal'] # 三线格的线种类

        self._text_model = [1,2] # 2种文字排版模式
        self._space_ratio2 = [0, 0.8] # 模式2空格率
        self._text_color = [(0, 0, 0), (0,0,255), (255, 0, 0), (180,180,180), (128,0,200)] # 文字颜色

        ##### text selector
        self.text_library_path = './Resource/Fonts' # 字体位置
        self.pun_ratio = 0.1 # 符号率
        self.space_ratio = 0.1 # 空格率（全页）
        self.space_ratio2 = 0.3 # 空白率（部分）

        ##### image enhance
        self._ifrotate = 0 # 0.7 # 旋转的概率
        self._angle = [-20, 20]
        self._ifperspective = 0 # 0 #0.7 # 透视的概率
        self._perspective_range = [0, 0.1] # 透视坐标变换比例
        self._ifnoise = 0.7 # 噪音的概率
        self._noise_std = [10,50] # 高斯噪声方差
        self._iflight = 0.7 # 光照的概率
        self._light_strength = [70,110] # 光照强度
        self._expand_range = [0.2, 0.3] # 扩大
        self._ifdesktop = 0.7 # 桌面的概率
        self._desktop_path = './Resource/Backgrounds'

def makedirs(x):
    for i in x: 
        if not os.path.exists(i): os.makedirs(i)

if __name__ == '__main__':
    start = 0
    end = 10001

    Parm = Parameters()
    text_selector = Text_Selector(Parm.pun_ratio, Parm.space_ratio, Parm.space_ratio2, Parm.text_library_path)
    

    path1 = Parm.save_path+'images/'+Parm.save_type
    path2 = Parm.save_path+'label_json/'+Parm.save_type
    path3 = Parm.save_path+'labels/'+Parm.save_type
    makedirs([path1, path2, path3]) 
    for i in tqdm(range(end-start)):
        page = MattsPage(Parm, text_selector)
        new_page = Image_Enhance(page)
        new_page.read_parm(Parm)
        new_page.image_enhance()
        matplotlib.image.imsave(path1 + f'/page_{i+start}.jpg', new_page.image)
        label_file = path2 + f'/page_{i+start}.json'
        # print(f"noise: {new_page.ifnoise} | light: {new_page.iflight} | perspective: {new_page.ifperspective} | rotate: {new_page.ifrotate} | desktop: {new_page.ifdesktop}" )
        with open(path2 + f'/page_{i+start}.json', 'w',encoding='utf-8') as file:
            json.dump(new_page.save_label,file, ensure_ascii=False)
        new_page.yolo_label.to_csv(path3+f'/page_{i+start}.txt', index=False, header=False, sep=' ', float_format='%.6f')

            