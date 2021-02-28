# -*- encoding: utf-8 -*-
'''
@Time        :2021/02/26 17:30:53
@Author      :Qu Yue
@File        :image_enhance.py
@Software    :Visual Studio Code
Introduction: Image enhancement
'''

#%%
%matplotlib qt5
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import random
from blocks import Parm, Text_Selector, MattsPage


#%% 
class Image_Enhance():
    def __init__(self, m_page):
        self.m_page = m_page
        self.ifrotate = True # 是否旋转
        self.ifperspective = True # 是否透视
        self.ifnoise = True # 是否噪音
        self.noise_std = 0.2 # 高斯噪声方差
        self.ifshadow = True # 是否阴影 
        self.change_size = ['expand', 'reduce', 'nochange']
        self.ifdesktop = True # 是否有桌面
        self.desktop_path = './Resource/Pictures'

        self.image = self.m_page.image
        self.blocks = self.m_page.blocks
        self.label_old = self.m_page.label
        self.label = self.m_page.label_e

        # self.Noise()
        # self.Rotate()


    def Noise(self):
        self.image = self.image.astype(np.float64)
        self.noise = np.round(np.random.normal(0, self.noise_std, self.image.shape))
        self.image += self.noise

        self.image[self.image<0] = 0
        self.image[self.image>255] = 255
        # self.image = self.image.astype(np.float64)
    
    def Rotate(self):
        hight, width = self.image.shape[:2]#self.m_page.width, self.m_page.hight
        self.angle = np.random.randint(-30, 30)
  
        hightNew = int(width * np.abs(np.sin(np.radians(self.angle))) + hight * np.abs(np.cos(np.radians(self.angle))))
        widthNew = int(width * np.abs(np.cos(np.radians(self.angle))) + hight * np.abs(np.sin(np.radians(self.angle))))
        matRotation = cv2.getRotationMatrix2D((width//2, hight//2), self.angle, 1)
        matRotation[0,2] += (widthNew - width)//2
        matRotation[1,2] += (hightNew - hight)//2
        self.image = cv2.warpAffine(self.image, matRotation,(widthNew,hightNew), borderValue=(0,0,0))
        
        for b in self.blocks:
            for block in b:
                new_corr = np.dot(matRotation,np.array([block.left_top+(1,), block.right_bottom+(1,)]).T)
                block.enhance(tuple(new_corr[:,0]), tuple(new_corr[:,1]))




#%%
if __name__ == '__main__':
    parm = Parm()
    text_selector = Text_Selector()
    page1 = MattsPage(parm, text_selector)
    new_page1 = Image_Enhance(page1)
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(new_page1.image)
    x = [new_page1.label[10]['corr'][0], new_page1.label[-1]['corr'][0]]
    y = [new_page1.label[10]['corr'][1], new_page1.label[-1]['corr'][1]]
    plt.scatter(x, y)

    plt.subplot(1,3,2)
    new_page1.Noise()
    plt.imshow(new_page1.image)
    plt.show()


    plt.subplot(1,3,3)
    new_page1.Rotate()
    plt.imshow(new_page1.image)

    x = [new_page1.m_page.label_e[10]['corr'][0], new_page1.m_page.label_e[-1]['corr'][0]]
    y = [new_page1.m_page.label_e[10]['corr'][1], new_page1.m_page.label_e[-1]['corr'][1]]
    plt.scatter(x,y)
    plt.show()
    
    
