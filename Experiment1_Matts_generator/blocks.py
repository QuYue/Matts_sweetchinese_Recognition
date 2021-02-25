#%%
# %matplotlib Qt5
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from draw_lines import drawline


#%%
class Parm():
    def __init__(self):
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

        self._threelines_type = ['dotted0','dotted1', 'normal']

class Block():
    # block_type: 'block', 'matts', 'grid'
    def __init__(self, left_top, right_bottom, image, block_type='matts', line_type='dotted1', line_gap=3):
        self.left_top = left_top # 左上角坐标 
        self.right_bottom = right_bottom # 右下角坐标
        self.image = image # 图片
        self.block_type = block_type
        self.line_type = line_type
        self.line_gap = line_gap

        self.top_y = self.left_top[1]
        self.bottom_y = self.right_bottom[1]
        self.left_x = self.left_top[0]
        self.right_x = self.right_bottom[0]
        self.middle_x = int((self.left_x + self.right_x)/2)
        self.middle_y = int((self.top_y + self.bottom_y)/2)
    
    def draw(self):
        cv2.rectangle(self.image, self.left_top, self.right_bottom, (0,0,0), 2)
        if self.block_type == 'matts':
            drawline(self.image, (self.middle_x, self.top_y), (self.middle_x, self.bottom_y), (0,0,0), 1, self.line_type, self.line_gap)
            drawline(self.image, (self.left_x, self.middle_y), (self.right_x, self.middle_y), (0,0,0), 1, self.line_type, self.line_gap)

        if self.block_type == 'grid':
            drawline(self.image, (self.middle_x, self.top_y), (self.middle_x, self.bottom_y), (0,0,0), 1, self.line_type, self.line_gap)
            drawline(self.image, (self.left_x, self.middle_y), (self.right_x, self.middle_y), (0,0,0), 1, self.line_type, self.line_gap)
            drawline(self.image, (self.left_x, self.top_y), (self.right_x, self.bottom_y), (0,0,0), 1, self.line_type, self.line_gap)
            drawline(self.image, (self.right_x, self.top_y), (self.left_x, self.bottom_y), (0,0,0), 1, self.line_type, self.line_gap)




class MattsPage():
    def __init__(self, parm):
        self.top_edge = np.random.uniform(parm._top_edge[0], parm._top_edge[1]) 
        self.bottom_edge = np.random.uniform(parm._bottom_edge[0], parm._bottom_edge[1]) 
        self.left_edge = np.random.uniform(parm._left_edge[0], parm._left_edge[1]) 
        self.right_edge = np.random.uniform(parm._right_edge[0], parm._right_edge[1]) 
        self.top_edge_line = np.random.randint(parm._top_edge_line[0], parm._top_edge_line[1]) 
        self.bottom_edge_line = np.random.randint(parm._bottom_edge_line[0], parm._bottom_edge_line[1]+1) 
        self.left_right_edge_line = np.random.randint(parm._left_right_edge_line[0], parm._left_right_edge_line[1]+1) 
        self.edge_distance = np.random.randint(parm._edge_distance[0], parm._edge_distance[1]+1) 

        self.block_size = np.random.randint(parm._block_size[0], parm._block_size[1]+1)
        self.block_row_num = np.random.randint(parm._block_row_num[0], parm._block_row_num[1]+1)
        self.block_column_num = np.random.randint(parm._block_column_num[0], parm._block_column_num[1]+1)
        self.block_type = random.sample(parm._block_type,1)[0]
        self.block_line_type = random.sample(parm._block_line_type,1)[0]
        self.block_line_gap0 = np.random.randint(parm._block_line_gap[0], parm._block_line_gap[1]+1)
        self.block_line_gap = int(self.block_size /self.block_line_gap0)
        self.threelines_type = random.sample(parm._threelines_type,1)[0]

        self.row_type = random.sample(parm._row_type, 1)[0]
        self.column_type = random.sample(parm._column_type, 1)[0]
        self.get_row_column_distance()
        self.row_line = random.sample(parm._row_line, 1)[0]
        self.column_line = random.sample(parm._column_line, 1)[0]
        if self.row_type == 'threelines' and self.column_line == 'no':
            self.column_line = random.sample(['no', 'edge','each','each'], 1)[0]


        self.hight_block = (self.block_size * self.block_row_num) + (self.row_distance * (self.block_row_num-1))
        if self.row_type == 'threelines':
            self.hight_block += self.row_distance
        self.hight = self.hight_block + self.top_edge_line + self.bottom_edge_line + (2 * self.edge_distance)
        self.width_block = (self.block_size * self.block_column_num) + (self.column_distance * (self.block_column_num-1))
        self.width = self.width_block + (2 * self.left_right_edge_line) + (2 * self.edge_distance)
        self.channel = 3

        self.get_img() # 生成纸
        self.get_edge_line() # 生成框
        self.get_edge() # 生成周围
        self.get_Blocks() # 生成格
        self.get_lines() # 生成横竖线

    def get_img(self):
        self.image = np.ones([self.hight, self.width, self.channel], np.uint8) * 255

    def get_edge_line(self):
        cv2.line(self.image, (int(self.left_right_edge_line/2),int(self.top_edge_line/2)), (self.width,int(self.top_edge_line/2)),(0,0,0),self.top_edge_line)
        cv2.line(self.image, (int(self.left_right_edge_line/2),int(self.top_edge_line/2)), (int(self.left_right_edge_line/2), self.hight),(0,0,0), self.left_right_edge_line)
        cv2.line(self.image, (self.width-int(self.left_right_edge_line/2), int(self.top_edge_line/2)), (self.width-int(self.left_right_edge_line/2),self.hight),(0,0,0),self.left_right_edge_line)
        cv2.line(self.image, (int(self.left_right_edge_line/2), self.hight-int(self.bottom_edge_line/2)), (self.width,self.hight-int(self.bottom_edge_line/2)),(0,0,0),self.bottom_edge_line)
        # cv2.rectangle(self.image, (0,0), (self.width,self.hight), (0,0,0), 10)

    def get_edge(self):
        self.top_edge_width = int(self.top_edge * self.hight)+5
        self.bottom_edge_width = int(self.bottom_edge * self.hight)+5
        self.left_edge_width = int(self.left_edge * self.width)+5
        self.right_edge_width = int(self.right_edge * self.width)+5

        self.page_hight = self.hight + self.top_edge_width + self.bottom_edge_width
        self.page_width = self.width + self.left_edge_width + self.right_edge_width

        self.image = np.pad(self.image, ((self.top_edge_width, self.bottom_edge_width), (self.left_edge_width, self.right_edge_width), (0, 0)), 'constant', constant_values=255)

    def get_row_column_distance(self):
        if self.row_type == 'nodistance':
            self.row_distance = 0
        elif self.row_type == 'threelines':
            self.row_distance = self.block_size -  np.random.randint(0, int(self.block_size/2)+1) 
        else:
            self.row_distance = np.random.randint(0, int(self.block_size/2)) 

        if self.column_type == 'nodistance':
            self.column_distance = 0
        else:
            self.column_distance = np.random.randint(0, int(self.block_size/3)) 

    def get_Blocks(self):
        self.left_top_x = np.arange(self.block_column_num) * (self.block_size+self.column_distance) + self.left_edge_width + self.left_right_edge_line + self.edge_distance
        self.left_top_y = np.arange(self.block_row_num) * (self.block_size+self.row_distance) + self.top_edge_width + self.top_edge_line + self.edge_distance
        
        if self.row_type == 'threelines':
            self.left_top_y += self.row_distance

        self.right_bottom_x = self.left_top_x + self.block_size
        self.right_bottom_y = self.left_top_y + self.block_size

        self.blocks = []
        for i in range(self.block_row_num):
            b = []
            for j in range(self.block_column_num):
                b.append(Block((self.left_top_x[j],self.left_top_y[i]), (self.right_bottom_x[j],self.right_bottom_y[i]), self.image, 
                                self.block_type, self.block_line_type, self.block_line_gap))
            self.blocks.append(b)

        for b in self.blocks:
            for block in b:
                block.draw()
            
    def get_lines(self):
        x1 = self.left_top_x[0]            
        y1 = self.left_top_y[0]
        if self.row_type == 'threelines':
            y1 -= self.row_distance
        x2 = self.right_bottom_x[-1]
        y2 = self.right_bottom_y[-1]

        if self.row_type == 'threelines':
            d = int(self.row_distance/3)
            for i in range(self.block_row_num):
                cv2.line(self.image, (x1,self.left_top_y[i]-self.row_distance), (x2,self.left_top_y[i]-self.row_distance),(0,0,0),2)
                cv2.line(self.image, (x1,self.left_top_y[i]), (x2,self.left_top_y[i]),(0,0,0),2)
                drawline(self.image, (x1,self.left_top_y[i]-(2*d)), (x2,self.left_top_y[i]-(2*d)), (0,0,0), 1, self.threelines_type, self.block_line_gap)
                drawline(self.image, (x1,self.left_top_y[i]-(1*d)), (x2,self.left_top_y[i]-(1*d)), (0,0,0), 1, self.threelines_type, self.block_line_gap)

        if self.row_line == 'edge':
            cv2.line(self.image, (x1-1,y1-1), (x2+1,y1-1),(0,0,0),4)
            cv2.line(self.image, (x1-1,y2+1), (x2+1,y2+1),(0,0,0),4)
        elif self.row_line == 'each':
            cv2.line(self.image, (x1-1,y1-1), (x2+1,y1-1),(0,0,0),4)
            cv2.line(self.image, (x1-1,y2+1), (x2+1,y2+1),(0,0,0),4)
            for i in range(self.block_row_num):
                if self.row_type ==  'nodistance':
                    cv2.line(self.image, (x1,self.left_top_y[i]), (x2,self.left_top_y[i]),(0,0,0),2)
                else:
                    cv2.line(self.image, (x1,self.left_top_y[i]), (x2,self.left_top_y[i]),(0,0,0),2)
                    cv2.line(self.image, (x1,self.right_bottom_y[i]), (x2,self.right_bottom_y[i]),(0,0,0),2)
        



        if self.column_line == 'edge':
            cv2.line(self.image, (x1-1,y1), (x1-1,y2), (0,0,0), 3)
            cv2.line(self.image, (x2+1,y1), (x2+1,y2), (0,0,0), 3)
        elif self.column_line == 'each':
            cv2.line(self.image, (x1-1,y1), (x1-1,y2), (0,0,0), 3)
            cv2.line(self.image, (x2+1,y1), (x2+1,y2), (0,0,0), 3)
            for i in range(self.block_column_num):
                if self.column_type ==  'nodistance':
                    cv2.line(self.image, (self.left_top_x[i], y1), (self.left_top_x[i], y2), (0,0,0), 2)
                else:
                    cv2.line(self.image, (self.left_top_x[i], y1), (self.left_top_x[i], y2), (0,0,0), 2)
                    cv2.line(self.image, (self.right_bottom_x[i], y1), (self.right_bottom_x[i], y2), (0,0,0), 2)





#%%
if __name__ == '__main__':
    parm = Parm()
    page1 = MattsPage(parm)
    page2 = MattsPage(parm)
    page3 = MattsPage(parm)
    page4 = MattsPage(parm)

#%%
    plt.figure(facecolor='gray')
    plt.subplot(1,4,1)
    plt.imshow(page1.image)
    plt.axis('off') 
    plt.subplot(1,4,2)
    plt.imshow(page2.image)
    plt.axis('off') 
    plt.subplot(1,4,3)
    plt.imshow(page3.image)
    plt.axis('off') 
    plt.subplot(1,4,4)
    plt.imshow(page4.image)
    plt.axis('off') 
    plt.show()

    num = 100
    
    for i in tqdm(range(num)):
        page = MattsPage(parm)
        matplotlib.image.imsave(f'./Pictures/page_{i}.jpg', page.image)
#%%