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
import pandas as pd
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as imgplt
import random
import time
import os
from blocks import Parm, Text_Selector, MattsPage
from draw_tools import image_overlay, make_pic_lights

#%% 
class Image_Enhance():
    def __init__(self, m_page):
        self.m_page = m_page
        self.ifrotate = True # 是否旋转
        self.angle = np.random.randint(-20, 20)
        self.ifperspective = True # 是否透视
        self.perspective_range = 0.1 # 透视坐标变换比例
        self.ifnoise = True # 是否噪音
        self.noise_std = 50 # 高斯噪声方差
        self.iflight = True # 是否光照 
        self.light_strength = 100
        self.change_size = ['expand', 'reduce', 'nochange']
        self.expand_range = 0.3 # 扩大
        self.ifdesktop = True # 是否有桌面
        self.desktop_path = './Resource/Backgrounds'

        self.image = self.m_page.image
        self.blocks = self.m_page.blocks
        self.size = self.m_page.size
    
    def proTrue(self, property):
        return True if np.random.rand()<property else False 

    def read_parm(self, parm):
        self.ifnoise = self.proTrue(parm._ifnoise) # 是否噪音
        self.noise_std = np.random.randint(parm._noise_std[0],parm._noise_std[1]) # 高斯噪声方差
        self.iflight = self.proTrue(parm._iflight) # 是否光照 
        self.light_strength = np.random.randint(parm._light_strength[0], parm._light_strength[1]) # 光照强度
        self.ifrotate = self.proTrue(parm._ifrotate) # 是否旋转
        self.angle = np.random.randint(parm._angle[0], parm._angle[1]) # 旋转角度
        self.ifperspective = self.proTrue(parm._ifperspective) # 是否透视
        self.perspective_range = np.random.uniform(parm._perspective_range[0], parm._perspective_range[1]) # 透视坐标变换比例
        self.ifdesktop = self.proTrue(parm._ifdesktop) # 是否有桌面
        # self.change_size = ['expand', 'reduce', 'nochange']
        self.expand_range = np.random.uniform(parm._expand_range[0],parm._expand_range[1]) # 扩大
        self.desktop_path = parm._desktop_path

    @property
    def label(self):
        return self.m_page.label_e

    @property
    def enhance(self):
        return {'noise': self.ifnoise, 
                'light': self.iflight, 
                'rotate': self.ifrotate,
                'perspective': self.ifperspective,
                'background': self.ifdesktop}

    @property
    def save_label(self):
        return {'enhance': self.enhance, 'label': self.label}
    
    @property
    def yolo_label(self):
        label = self.save_label['label']
        new_label = []
        shape = self.image.shape
        for i in label:
            _class = 0
            if i['text'] == ' ':
                _class = 0
            elif i['text'] in "。，、；！？“”（）：+-=%":
                _class = 2
            else:
                _class = 1
            x1 = i['corr'][0]
            x2 = i['corr'][2]
            y1 = i['corr'][1]
            y2 = i['corr'][3]
            x_center = ((x1 + x2) / 2) / shape[1]
            y_center = ((y1 + y2) / 2) / shape[0]
            width = (x2 - x1) / shape[1]
            height = (y2 - y1) / shape[0]
            new_label.append([_class, x_center, y_center, width, height])
        yolo_label = pd.DataFrame(new_label, columns=['class','x_center','y_center', 'width','height'])
        return yolo_label

    def Alpha(self, image):
        alpha_channel = np.ones([image.shape[0], image.shape[1], 1]).astype(np.uint8) * 255
        image = np.concatenate([image[...,:3], alpha_channel], 2)
        return image

    def Noise(self):
        self.image = self.image.astype(np.int64)
        self.noise = np.round(np.random.normal(0, self.noise_std, [self.image.shape[0],self.image.shape[1],3])).astype(np.int64)
        self.image[...,:3] += self.noise

        self.image[self.image<0] = 0
        self.image[self.image>255] = 255
        self.image = self.image.astype(np.uint8)

    def Perspective(self):
        corr_change = np.random.uniform(0,self.perspective_range,[4,2])
        hight, width, channels = self.image.shape
        p1 = np.float32([[0,0], [width-1,0], [0,hight-1], [width-1, hight-1]])
        p2 = np.float32([[corr_change[0,0]*width, corr_change[0,1]*hight], [(1-corr_change[1,0])*width, corr_change[1,1]*hight],
                         [corr_change[2,0]*width, (1-corr_change[2,1])*hight], [(1-corr_change[3,0])*width, (1-corr_change[3,1])*hight]])
        Mat = cv2.getPerspectiveTransform(p1,p2)
        self.image = cv2.warpPerspective(self.image, Mat, (width, hight))

        for b in self.blocks:
            for block in b:
                new_corr = cv2.perspectiveTransform(np.array([[block.left_top_e, block.right_bottom_e]], dtype=np.float32), Mat)
                block.enhance(tuple(new_corr[0,0,:]), tuple(new_corr[0,1,:]))
    
    def Rotate(self):
        hight, width = self.image.shape[:2]#self.m_page.width, self.m_page.hight
        hightNew = int(width * np.abs(np.sin(np.radians(self.angle))) + hight * np.abs(np.cos(np.radians(self.angle)))) + 60
        widthNew = int(width * np.abs(np.cos(np.radians(self.angle))) + hight * np.abs(np.sin(np.radians(self.angle)))) + 60
        mat = cv2.getRotationMatrix2D((width//2, hight//2), self.angle, 1)
        mat[0,2] += (widthNew - width)//2 + 10
        mat[1,2] += (hightNew - hight)//2 + 10
        self.image = cv2.warpAffine(self.image, mat,(widthNew,hightNew), borderValue=(0,0,0,0))
        
        for b in self.blocks:
            for block in b:
                new_corr = np.dot(mat,np.array([block.left_top_e+(1,), block.right_bottom_e+(1,)]).T)
                block.enhance(tuple(new_corr[:,0]), tuple(new_corr[:,1]))

    def Expand(self):
        hight, width = self.image.shape[:2]
        self.top_expand = int(np.random.uniform(0, self.expand_range)*hight)
        self.bottom_expand = int(np.random.uniform(0, self.expand_range)*hight)
        self.left_expand = int(np.random.uniform(0, self.expand_range)*width)
        self.right_expand = int(np.random.uniform(0, self.expand_range)*width)

        self.image = np.pad(self.image, ((self.top_expand, self.bottom_expand), (self.left_expand, self.right_expand), (0, 0)), 'constant', constant_values=0)
        for b in self.blocks:
            for block in b:
                new_corr = np.array([block.left_top_e, block.right_bottom_e]) + [self.left_expand, self.top_expand]
                block.enhance(tuple(new_corr[0,:]), tuple(new_corr[1,:]))

    def Light(self):
        self.image = make_pic_lights(self.image, self.light_strength)

    def Backgrounds(self):
        file_name_list = os.listdir(self.desktop_path)
        file_name = random.sample(file_name_list, 1)[0]
        self.desktop = imgplt.imread(self.desktop_path+'/'+file_name)
        self.desktop = cv2.resize(self.desktop, (self.image.shape[1], self.image.shape[0]))
        self.desktop = self.Alpha(self.desktop)
        self.image = image_overlay(self.image, self.desktop)

    def Cut(self):
        self.top_cut = np.random.randint(1, self.top_expand + 10)
        self.bottom_cut = np.random.randint(1, self.bottom_expand + 10)
        self.left_cut = np.random.randint(1, self.left_expand + 10)
        self.right_cut = np.random.randint(1, self.right_expand + 10)

        self.image = self.image[self.top_cut: -self.bottom_cut, self.left_cut: -self.right_cut, :3]

        for b in self.blocks:
            for block in b:
                new_corr = np.array([block.left_top_e, block.right_bottom_e]) - [self.left_cut, self.top_cut]
                block.enhance(tuple(new_corr[0,:]), tuple(new_corr[1,:]))

    
    def image_enhance(self):
        self.image = self.Alpha(self.image)
        if self.ifnoise:
            self.Noise()
        if self.iflight:
            self.Light()
        if self.ifperspective:
            self.Perspective()
        if self.ifrotate:
            self.Rotate()
        if self.ifdesktop:
            self.Expand()
            self.Backgrounds()
            self.Cut()
        self.image = self.image[:,:,:3]

#%%
if __name__ == '__main__':
    parm = Parm()
    text_selector = Text_Selector(0.1, 0.1, 0.3, parm.text_library_path)
    page1 = MattsPage(parm, text_selector)
    new_page1 = Image_Enhance(page1)
    plt.figure()
    plt.subplot(2,4,1)
    plt.imshow(new_page1.image)
    x = [new_page1.label[10]['corr'][0], new_page1.label[-1]['corr'][0]]
    y = [new_page1.label[10]['corr'][1], new_page1.label[-1]['corr'][1]]
    plt.scatter(x, y)
    plt.axis('off') 
    plt.title('Original')

    plt.subplot(2,4,2)
    start = time.time()
    new_page1.Noise()
    end = time.time()
    plt.imshow(new_page1.image)
    plt.axis('off') 
    plt.title(f'Noise {end-start :^ 8.4f}s')

    plt.subplot(2,4,3)
    start = time.time()
    new_page1.Light()
    end = time.time()
    plt.imshow(new_page1.image)
    plt.axis('off') 
    plt.title(f'Light {end-start :^ 8.4f}s')

    plt.subplot(2,4,4)
    start = time.time()
    new_page1.Perspective()
    end = time.time()
    x = [new_page1.label[10]['corr'][0], new_page1.label[-1]['corr'][0]]
    y = [new_page1.label[10]['corr'][1], new_page1.label[-1]['corr'][1]]
    plt.imshow(new_page1.image)
    plt.scatter(x,y)
    plt.axis('off') 
    plt.title(f'Perspective {end-start :^ 8.4f}s')

    plt.subplot(2,4,5)
    start = time.time()
    new_page1.Rotate()
    end = time.time()
    plt.imshow(new_page1.image)
    x = [new_page1.label[10]['corr'][0], new_page1.label[-1]['corr'][0]]
    y = [new_page1.label[10]['corr'][1], new_page1.label[-1]['corr'][1]]
    plt.scatter(x,y)
    plt.axis('off') 
    plt.title(f'Rotate {end-start :^ 8.4f}s')

    plt.subplot(2,4,6)
    start = time.time()
    new_page1.Expand()
    end = time.time()
    plt.imshow(new_page1.image)
    x = [new_page1.label[10]['corr'][0], new_page1.label[-1]['corr'][0]]
    y = [new_page1.label[10]['corr'][1], new_page1.label[-1]['corr'][1]]
    plt.scatter(x,y)
    plt.axis('off') 
    plt.title(f'Expand {end-start :^ 8.4f}s')

    plt.subplot(2,4,7)
    start = time.time()
    new_page1.Backgrounds()
    end = time.time()
    plt.imshow(new_page1.image)
    x = [new_page1.label[10]['corr'][0], new_page1.label[-1]['corr'][0]]
    y = [new_page1.label[10]['corr'][1], new_page1.label[-1]['corr'][1]]
    plt.scatter(x,y)
    plt.axis('off') 
    plt.title(f'Backgrounds {end-start :^ 8.4f}s')

    plt.subplot(2,4,8)
    start = time.time()
    new_page1.Cut()
    end = time.time()
    plt.imshow(new_page1.image)
    x = [new_page1.label[10]['corr'][0], new_page1.label[-1]['corr'][0]]
    y = [new_page1.label[10]['corr'][1], new_page1.label[-1]['corr'][1]]
    plt.scatter(x,y)
    plt.axis('off') 
    plt.title(f'Cut {end-start :^ 8.4f}s')

    plt.show()


