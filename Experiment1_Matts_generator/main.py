# -*- encoding: utf-8 -*-
'''
@Time        :2021/03/17 19:34:44
@Author      :Qu Yue
@File        :main.py
@Software    :Visual Studio Code
Introduction:  12
'''
#%%
# %matplotlib qt5
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as imgplt
import random
import time
import json
from tqdm import tqdm
from blocks import Text_Selector, MattsPage
from draw_tools import image_overlay, make_pic_lights
from image_enhance import Image_Enhance
#%%

class Parm():
    def __init__(self):
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
        self._ifrotate = 0.7 # 旋转的概率
        self._angle = [-20, 20]
        self._ifperspective = 0.7 # 透视的概率
        self._perspective_range = [0, 0.1] # 透视坐标变换比例
        self._ifnoise = 0.7 # 噪音的概率
        self._noise_std = [10,50] # 高斯噪声方差
        self._iflight = 0.7 # 光照的概率
        self._light_strength = [70,110] # 光照强度
        self._expand_range = [0.2, 0.3] # 扩大
        self._ifdesktop = 0.7 # 桌面的概率
        self._desktop_path = './Resource/Backgrounds'

if __name__ == '__main__':
    num=20
    parm = Parm()
    text_selector = Text_Selector(parm.pun_ratio, parm.space_ratio, parm.space_ratio2, parm.text_library_path)

    for i in tqdm(range(num)):
        page = MattsPage(parm, text_selector)
        new_page = Image_Enhance(page)
        new_page.read_parm(parm)
        new_page.image_enhance()
        matplotlib.image.imsave(f'../../Datasets/Dataset1/data/page_{i}.jpg', new_page.image)
        label_file=f'../../Datasets/Dataset1/label/page_{i}.json'
        with open(label_file,'w') as file:
            json.dump(new_page.label,file)