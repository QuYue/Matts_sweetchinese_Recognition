# -*- encoding: utf-8 -*-
'''
@Time        :2021/02/26 17:30:53
@Author      :Qu Yue
@File        :image_enhance.py
@Software    :Visual Studio Code
Introduction: Image enhancement
'''

#%%
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import random


#%% 
class Image_Enhance():
    def __init__(self, m_page):
        self.m_page = m_page
        self.rotate = True # 是否旋转
        self.perspective = True # 是否透视
        self.noise = True # 是否噪音
        self.shadow = True # 是否阴影 
        self.change_size = ['expand', 'reduce', 'nochange']
        self.desktop = True # 是否有桌面
        self.desktop_path = './Resource/Pictures'

#%%
if __name__ == '__main__':
    a = Image_Enhance()
    
