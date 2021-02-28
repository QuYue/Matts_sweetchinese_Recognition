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
        self.rotate = True # 是否旋转
        self.perspective = True # 是否透视
        self.noise = True # 是否噪音
        self.shadow = True # 是否阴影 
        self.change_size = ['expand', 'reduce', 'nochange']
        self.desktop = True # 是否有桌面
        self.desktop_path = './Resource/Pictures'

        self.Rotate()
    
    def Rotate(self):
        hight, width = self.m_page.image.shape[:2]#self.m_page.width, self.m_page.hight
        self.angle = np.random.randint(-30, 30)
  
        hightNew = int(width * np.abs(np.sin(np.radians(self.angle))) + hight * np.abs(np.cos(np.radians(self.angle))))
        widthNew = int(width * np.abs(np.cos(np.radians(self.angle))) + hight * np.abs(np.sin(np.radians(self.angle))))
        matRotation = cv2.getRotationMatrix2D((width//2, hight//2), self.angle, 1)
        matRotation[0,2] += (widthNew - width)//2
        matRotation[1,2] += (hightNew - hight)//2
        self.imgRotation = cv2.warpAffine(self.m_page.image, matRotation,(widthNew,hightNew), borderValue=(0,0,0))
        
        x = self.m_page.label[0]['corr'][0]
        y = self.m_page.label[0]['corr'][1]
        print(matRotation.shape)
        self.Q = np.dot(matRotation,np.array([[x],[y],[1]]))
        for b in self.m_page.blocks:
            for block in b:
                left_top = np.dot(MatRotation,np.array([left_top+[1], right_bottem+[1]]).T)



#%%
if __name__ == '__main__':
    parm = Parm()
    text_selector = Text_Selector()
    page1 = MattsPage(parm, text_selector)
    new_page1 = Image_Enhance(page1)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(new_page1.m_page.image)

    x = [new_page1.m_page.label[0]['corr'][0], new_page1.m_page.label[1]['corr'][0]]
    y = [new_page1.m_page.label[0]['corr'][1], new_page1.m_page.label[1]['corr'][1]]
    plt.scatter(x, y)
    plt.subplot(1,2,2)
    plt.imshow(new_page1.imgRotation)
    plt.scatter(new_page1.Q[0], new_page1.Q[1])
    plt.show()
    
    
