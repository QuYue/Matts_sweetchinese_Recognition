import cv2
import numpy as np
import random
import math
from PIL import Image, ImageDraw, ImageFont
import csv
from math import *
import  os
import shutil

'''需要修改'''
amount_start = 0
amount_end = 50
#生成图片的数量,编号从amount_start到amount_end

'''可修改'''
pic_size = (1680, 1920)
#生成的图片的大小，默认为1680*1920

'''需要修改'''
file_path = 'D:\\projects\\pic\\case\\'
#生成结果文件的路径


distance = 10
#单位间隔距离，田字格行间间距的最小单位，实际使用时可以表示成随机整数倍，达到不同间隔的目的

outline_color=[(255,140,0),(255,20,147),(205,92,92),(255,127,80)]
#边框颜色待选列表，可以增加新的待选颜色


#生成图片函数
def make_pics():
    pic_configs= []
    '''设置了一个列表用来保存每张图片的信息
    每张图片的信息包括 图片中的田字格的行数，列数，田字格中的每个方框的高度
    '''
    for i in range(amount_start, amount_end):
        single_pic= []

        '''可修改'''
        column_amount = random.randint(5, 12)
        dimension = min(2*column_amount,10)
        row_amount =  random.randint(5,dimension)
        #确定田字格的行数和列数，出于美观的考虑设置了如上的比例

        '''可修改'''
        h = random.randint(100,120)
        #每个方格的大小设置


        single_pic.append(row_amount)
        single_pic.append(column_amount)
        single_pic.append(h)
        pic_configs.append(single_pic)
        #将每张图片的信息组装成一个列表


        img = Image.new('RGB', (h*column_amount,(h+4*distance)*row_amount), (255, 255, 255))
        img.save(file_path + str(i) + '.jpg')  # 生成空白图像，第一个参数为颜色的模式；第二个参数为图片的大小；第三个参数指的是图像的颜色；
        #该图片是生成的田字格图片


        img = Image.new('RGB', pic_size, (255, 255, 255))
        img.save(file_path + "fushion" + str(i) + '.jpg')
        #该图片是生成的白底图片，相当于田字格图片外面的那层“壳”

    return pic_configs

'''从2500个待选汉字中随机生成不同文字'''
def random_char():
    with open("library.txt", "r", encoding='UTF-8') as f:
        data = f.readline()
    # data中保存了2500个字
    index = random.randint(0, 2499)
    return data[index]

'''画虚线函数，需要传入起始点和终点的坐标'''
def draw_transparent_line(start_x,start_y,end_x,end_y,image):

    column_num = end_x-start_x
    for x in range(start_x, column_num, 3):
        image[start_y, x] = 255 - image[end_y,x]
    return image

'''画虚线函数，和1的区别是一个画的是横向的，一个画的是纵向的'''
def draw_transparent_line2(start_x,start_y,end_x,end_y,image):

    row_num = end_y-start_y
    for x in range(start_y, row_num, 3):
        image[x, start_x] = 255 - image[x,end_x]
    return image

'''在指定图片上根据所给位置添加文字'''
# image: 图片  text：要添加的文本 font：字体
def add_text_to_image(image, position, text,font):
    rgba_image = image.convert('RGBA')
    text_overlay = Image.new('RGBA', rgba_image.size, (255, 255, 255, 0))
    image_draw = ImageDraw.Draw(text_overlay)

    # 设置文本颜色和透明度
    image_draw.text(position, text, font=font, fill=(112, 128, 144, 180))
    image_with_text = Image.alpha_composite(rgba_image, text_overlay)

    return image_with_text


def draw_lines(pic_configs):

    position = []#用来表示所有图片中每一行田字格的起始位置

    for x in range(amount_start, amount_end):

        image = cv2.imread(file_path+str(x) + ".jpg")#读取生成的图片

        sp = image.shape
        height = sp[0]
        width=sp[1]
        #获取生成的空白图片宽，高

        pic_info = pic_configs[x-amount_start]
        row_amount = pic_info[0]
        column_amount = pic_info[1]
        #获取该图片对应的信息，包括需要生成田字格的行数，列数，和田字格的宽度

        h = pic_info[2]
        #print(row_amount, column_amount,h)
        half_h = int(h/2)

        random_num = random.randint(0,len(outline_color)-1)
        cv2.rectangle(image, (0,0), (width, height),(0,0,0),2)
        # 绘制田字格外框


        old_position_Y = 0
        line_position = []#用一个列表来保存每一行田字格的起始位置

        for i in range(0, row_amount):
            line_position.append(old_position_Y)
            for j in range(0, column_amount):
                draw_1 = cv2.rectangle(image, (h * j, old_position_Y), (h * j + h, old_position_Y + h),outline_color[random_num],2)
        #绘制每个田字格

            random_num_new =random.randint(1,4)#行间间隔设置，表示为最小单位的整数倍

            old_position_Y = old_position_Y+h+random_num_new*distance#更新下一行田字格的起始位置
            if random_num_new>1:
                for m in range(0,random_num_new):
                    rand_num_test = random.randint(1,4)
                    if(m*rand_num_test%2==0):#相邻两行田字格中间的线实虚间隔排列，奇数为实线，偶数为虚线
                        cv2.line(image,(0,old_position_Y-distance*m),(h*column_amount,old_position_Y-distance*m),outline_color[random_num],1)
                        draw_transparent_line(0,old_position_Y-distance*m, h*column_amount,old_position_Y-distance*m,image)
                    else:
                        cv2.line(image, (0, old_position_Y - distance * m),(h * column_amount, old_position_Y - distance * m), outline_color[random_num], 1)

                #cv2.line(image,(0,(h + 4*distance) * i + h+5),(h*column_amount,(h + 4*distance) * i + h+5),1)

        for n in range(0,len(line_position)):
            cv2.line(image, (0, line_position[n]+half_h), (h * column_amount, line_position[n]+half_h),outline_color[random_num], 1)
            draw_transparent_line(0, line_position[n]+half_h,h * column_amount, line_position[n]+half_h,image)

        for q in range(0,column_amount):
            cv2.line(image, (q*h+half_h, 0), (q*h+half_h, line_position[-1]+h), outline_color[random_num],1)
            draw_transparent_line2(q*h+half_h, 0,q*h+half_h, line_position[-1]+h,image)
        #分别绘制行虚线，列虚线

        position.append(line_position)#保存该行的起始位置
        #make_pic_lights(image)
        cv2.imwrite(file_path + str(x) + '.jpg', image)#覆盖之前的空白图片

    return position


def fill_words(position,pic_configs):
    lines_length = len(position)
    # print(lines_length)
    # print(position)
    words_num = 0
    words_config=[]
    words_list=[]

    for x in range(0,lines_length):
        im_after = Image.open(file_path + str(x+amount_start) + '.jpg')
        pic_info = pic_configs[x]
        row_amount = pic_info[0]
        column_amount = pic_info[1]
        h = pic_info[2]
        half_h = int(h / 2)

        '''可修改'''
        char_distance = int(h*(1/6))
        #char_distance 表示为生成的文字距离田字格上边框和左边框的相对距离

        '''可修改'''
        font_size = int(h*(2/3))
        font = ImageFont.truetype('D:\projects\\font\kaiti.ttf', font_size)#设置文字的字体

        words_num  = random.randint(1,row_amount*column_amount)#确定要填充的文字的数量
        block_num = random.randint(0,words_num%10)#确定要覆盖的色块的数量

        block_position = []

        if block_num>0:
            block_position = random.sample(range(0, words_num), block_num)#随机生成色块位置
        # print(words_num,block_num)
        # print(block_position)



        count = 0
        words_detail=[]#用来保存一张图片中的所有文字

        for i in range(0, row_amount):
            for j in range(0, column_amount):
                random_character = random_char()
                words_detail.append(random_character)#添加文字信息
                im_after = add_text_to_image(im_after, (j * h + char_distance, position[x][i] + char_distance),random_character,font)#添加在图片对应位置上
                count = count + 1
                if (count>=words_num):#加判断是因为words_num<=行列之积
                    break
            if (count>=words_num):
                break


        block_position_new = []
        if len(block_position)>0 and (len(block_position)<words_num):
            for n in range(0,len(block_position)):#添加色块
                block_position_new.append(block_position[n])
                if(block_position[n]<column_amount):
                    position1_x = block_position[n]*h
                    position1_y = 0
                    position2_x = position1_x+h
                    position2_y = position1_y+h

                    draw = ImageDraw.Draw(im_after)
                    '''可修改'''
                    draw.rectangle((position1_x, position1_y, position2_x, position2_y), '#879f65')#设置遮挡色块的颜色
                else:
                    line_no = int(block_position[n]/column_amount)
                    column_no = block_position[n]-line_no*column_amount
                    #get the right line_no and column_no
                    position1_x = column_no*h
                    position1_y = position[x][line_no]
                    position2_x = position1_x + h
                    position2_y = position1_y + h
                    draw = ImageDraw.Draw(im_after)
                    draw.rectangle((position1_x, position1_y, position2_x, position2_y), '#879f65')

        words_info = [words_num, block_position_new]#保存图片中的相关文字信息，包含文字的数量，每个文字的位置
        words_config.append(words_info)
        words_list.append(words_detail)

        im_after = im_after.convert('RGB')
        im_after.save(file_path + str(x+amount_start) + '.jpg')#覆盖原来的图片

    return words_config,words_list

#设置图片光照的具体实现
def make_pic_lights(img):
    # 读取原始图像
    # 获取图像行和列
    rows, cols = img.shape[:2]

    '''可修改'''
    centerX = random.randint(0,rows)# 设置中心点
    centerY = random.randint(0,cols)

    print(centerX, centerY)
    radius = min(rows, cols)/4
    print(radius)


    '''可修改'''
    strength = random.randint(0,90)# 设置光照强度

    # 图像光照特效
    for i in range(rows):
        for j in range(cols):
            # 计算当前点到光照中心距离(平面坐标系中两点之间的距离)
            distance = math.pow((centerY - j), 2) + math.pow((centerX - i), 2)
            # 获取原始图像
            B = img[i, j][0]
            G = img[i, j][1]
            R = img[i, j][2]

            result = (int)(strength * (1.0 - math.sqrt(distance) / radius))

            B = img[i, j][0] + result
            G = img[i, j][1] + result
            R = img[i, j][2] + result
            # 判断边界 防止越界
            B = min(255, max(0, B))
            G = min(255, max(0, G))
            R = min(255, max(0, R))
            img[i, j] = np.uint8((B, G, R))
    print("finished")

def rotate_bound_white_bg(image, degree,x):

    height, width = image.shape[:2]
    #获取宽高
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    #计算新图片的宽高
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
    #对图片以中心进行无缩放旋转

    matRotation[0, 2] += (widthNew - width) // 2
    matRotation[1, 2] += (heightNew - height) // 2
    #设置平移量

    imgRotation = cv2.warpAffine(image, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))

    with open(file_path+'data'+str(x)+'.csv', 'r',encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]

    os.remove(file_path + 'data' + str(x) + '.csv')
    f = open(file_path + 'result_data' + str(x) + '.csv', 'w', newline='', encoding='utf-8-sig')
    # 2. 基于文件对象构建 csv写入对象
    csv_writer = csv.writer(f)
    # 3. 构建列表头
    csv_writer.writerow(["字符", "方格宽度", "左上", "右上", "右下", "左下"])
    for n in range(1,len(rows)):
        positions = []
        written_position1 = ()
        written_position2 = ()
        written_position3 = ()
        written_position4 = ()
        for m in range(2,6):
            point = rows[n][m]
            position = transferacord(point)
            written_position = np.dot(matRotation,np.array([[position[0]], [position[1]], [1]]))
            local_position = (int(written_position[0][0]),int(written_position[1][0]))
            positions.append(local_position)
        for m in (0,len(positions)):
            written_position1 = positions[0]
            written_position2 = positions[1]
            written_position3 = positions[2]
            written_position4 = positions[3]
        written_char = rows[n][0]
        written_height = int(rows[n][1])
        csv_writer.writerow([written_char, written_height, written_position1, written_position2, written_position3, written_position4])
    f.close()

    with open(file_path+'keypoint'+str(x)+'.csv', 'r',encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]

    positions = []
    for m in range(1, 6):
        point = rows[1][m]
        position = transferacord(point)
        written_position = np.dot(matRotation, np.array([[position[0]], [position[1]], [1]]))
        local_position = (int(written_position[0][0]), int(written_position[1][0]))
        positions.append(local_position)

    written_position1 = positions[0]
    written_position2 = positions[1]
    written_position3 = positions[2]
    written_position4 = positions[3]
    written_position5 = positions[4]
    print(written_position1, written_position2, written_position3, written_position4,written_position5)
    os.remove(file_path+'keypoint'+str(x)+'.csv')
    f = open(file_path + 'keypoint' + str(x) + '.csv', 'w', newline='', encoding='utf-8-sig')
    # 2. 基于文件对象构建 csv写入对象
    csv_writer = csv.writer(f)
    # 3. 构建列表头
    csv_writer.writerow(["图片序号", "左上", "右上", "右下", "左下", "图片中点"])
    # 4. 写入csv文件内容
    csv_writer.writerow(
        [x, written_position1, written_position2, written_position3, written_position4,written_position5])
    f.close()

    return imgRotation

def twist_pic():
    for count in range(amount_start,amount_end):
        img = cv2.imread(file_path + "fushion"+str(count)+".jpg")

        '''可修改'''
        angle = random.randint(1,45)#随机设置扭曲的基准角度

        print("count:"+str(count),"angle"+str(angle))
        rows, cols = img.shape[:2]
        img_output = np.zeros(img.shape, dtype=img.dtype)

        with open(file_path + 'data' + str(count) + '.csv', 'r', encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            datas = [row for row in reader]

        for i in range(rows):
            for j in range(cols):

                offset_x = int(angle * math.sin(2 * 3.14 * i / cols))
                offset_y = int(angle * math.cos(2 * 3.14 * j / cols))
                if i+offset_y < rows and j+offset_x < cols:
                    img_output[i,j] = img[i+offset_y,j+offset_x]
                else:
                    img_output[i, j] = 0

        os.remove(file_path + 'data' + str(count) + '.csv')
        f = open(file_path + 'result_data' + str(count) + '.csv', 'w', newline='', encoding='utf-8-sig')
        # 2. 基于文件对象构建 csv写入对象
        csv_writer = csv.writer(f)
        # 3. 构建列表头
        csv_writer.writerow(["字符", "方格宽度", "左上", "右上", "右下", "左下"])
        for n in range(1, len(datas)):
            positions = []
            written_position1 = ()
            written_position2 = ()
            written_position3 = ()
            written_position4 = ()

            for m in range(2, 6):
                point = datas[n][m]
                position = transferacord(point)
                x = position[0]
                y = position[1]

                '''可修改'''
                offset_x = int(angle * math.sin(2 * 3.14 * y / cols))#图片扭曲的程度
                offset_y = int(angle * math.cos(2 * 3.14 * x / cols))


                x = x-offset_x
                y = y-offset_y
                positions.append((x,y))


            written_position1 = positions[0]
            written_position2 = positions[1]
            written_position3 = positions[2]
            written_position4 = positions[3]

            written_char = datas[n][0]
            written_height = int(datas[n][1])
            csv_writer.writerow([written_char, written_height, written_position1, written_position2, written_position3,
                                 written_position4])

        os.remove(file_path+str(count)+'.jpg')
        os.remove(file_path+'fushion'+str(count)+'.jpg')
        cv2.imwrite(file_path+"result"+str(count)+'.jpg',img_output)

        with open(file_path + 'keypoint' + str(count) + '.csv', 'r', encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]

        positions = []
        for m in range(1, 6):
            point = rows[1][m]
            position = transferacord(point)
            x = position[0]
            y = position[1]
            offset_x = int(angle * math.sin(2 * 3.14 * y / cols))
            offset_y = int(angle * math.cos(2 * 3.14 * x / cols))
            x = x - offset_x
            y = y - offset_y
            positions.append((x, y))

        written_position1 = positions[0]
        written_position2 = positions[1]
        written_position3 = positions[2]
        written_position4 = positions[3]
        written_position5 = positions[4]
        print(written_position1, written_position2, written_position3, written_position4, written_position5)
        os.remove(file_path + 'keypoint' + str(count) + '.csv')
        f = open(file_path + 'keypoint' + str(count) + '.csv', 'w', newline='', encoding='utf-8-sig')
        # 2. 基于文件对象构建 csv写入对象
        csv_writer = csv.writer(f)
        # 3. 构建列表头
        csv_writer.writerow(["图片序号", "左上", "右上", "右下", "左下", "图片中点"])
        # 4. 写入csv文件内容
        csv_writer.writerow(
            [x, written_position1, written_position2, written_position3, written_position4, written_position5])
        f.close()
        img_output = cv2.circle(img_output, written_position1, 1, (255, 0, 0), 1, 8, 0)
        img_output = cv2.circle(img_output, written_position2, 1, (255, 0, 0), 1, 8, 0)
        img_output = cv2.circle(img_output, written_position3, 1, (255, 0, 0), 1, 8, 0)
        img_output = cv2.circle(img_output, written_position4, 1, (255, 0, 0), 1, 8, 0)
        img_output = cv2.circle(img_output, written_position5, 1, (255, 0, 0), 1, 8, 0)


def make_lights():
    for x in range(amount_start,amount_end):
        image = cv2.imread(file_path + "result"+str(x)+".jpg")
        make_pic_lights(image)#光照效果的具体实现
        cv2.imwrite(file_path + "result"+str(x) + '.jpg', image)

def rotate_pic():
    for x in range(amount_start,amount_end):
        img = cv2.imread(file_path + "fushion"+str(x)+".jpg")

        '''可修改'''
        random_angle = random.randint(0,10)#随机设置图片旋转的角度

        imgRotation = rotate_bound_white_bg(img, random_angle,x)#旋转图片的具体实现

        os.remove(file_path+'fushion'+str(x)+'.jpg')
        os.remove(file_path+str(x)+'.jpg')
        cv2.imwrite(file_path + "result"+str(x) + '.jpg', imgRotation)

def fushion_pics(pic_configs):
    for x in range(amount_start,amount_end):
        pic_info = pic_configs[x-amount_start]
        h = pic_configs[x-amount_start][2]
        img = Image.open(file_path + str(x) + '.jpg')
        oriImg = Image.open(file_path +"fushion"+ str(x) + '.jpg')
        oriImg.paste(img, (h,h))
        oriImg.save(file_path +"fushion" + str(x) + ".jpg")

def transferacord(point):
    point = point.replace('(', '')
    point = point.replace(')', '')
    point = point.split(',', 1)
    position_x = int(point[0])
    position_y = int(point[1])
    position = (position_x, position_y)
    return position

def make_data(pic_configs,position,words_config,words_list):

    for x in range(amount_start,amount_end):
        img = cv2.imread(file_path + "fushion"+str(x)+".jpg")



        pic_data = {}
        pic_info = pic_configs[x-amount_start]
        row_amount = pic_info[0]
        column_amount = pic_info[1]
        h = pic_configs[x-amount_start][2]

        height = (h + 4 * distance) * row_amount
        width = h * column_amount

        words_num = words_config[x-amount_start][0]
        block_position = words_config[x-amount_start][1]
        words_detail = words_list[x-amount_start]
        #获取图片的宽度，高度，田字格单位大小，田字格内填充的文字，文字的所在的位置

        for j in range(0,len(position[x-amount_start])):
            position[x-amount_start][j] = position[x-amount_start][j]+h

        for i in range(0,words_num):
            if i not in block_position:
                key = words_detail[i]
                if i<column_amount:
                    position1_x = position[x-amount_start][0]+i*h
                    position1_y = position[x-amount_start][0]
                    position2_x = position1_x+h
                    position2_y = position1_y
                    position3_x=position2_x
                    position3_y=position2_y+h
                    position4_x = position1_x
                    position4_y = position1_y+h
                    position1=(position1_x,position1_y)
                    position2=(position2_x,position2_y)
                    position3=(position3_x,position3_y)
                    position4 = (position4_x,position4_y)
                    location = [position1,position2,position3,position4]
                    pic_data[key] = location
                else:
                    line_no = int(i/column_amount)
                    column_no = i - line_no * column_amount
                    position1_x = h+column_no * h
                    position1_y = position[x-amount_start][line_no]
                    position2_x = position1_x + h
                    position2_y = position1_y
                    position3_x = position2_x
                    position3_y = position2_y + h
                    position4_x = position1_x
                    position4_y = position1_y + h
                    position1 = (position1_x, position1_y)
                    position2 = (position2_x, position2_y)
                    position3 = (position3_x, position3_y)
                    position4 = (position4_x, position4_y)
                    location = [position1, position2, position3, position4,line_no,column_no]
                    pic_data[key] = location

        #按照从左至右，从上至下的顺序依次组装完成

        f  = open(file_path+'data'+str(x)+'.csv','w',newline='',encoding='utf-8-sig')
        # 2. 基于文件对象构建 csv写入对象
        csv_writer = csv.writer(f)
        # 3. 构建列表头
        csv_writer.writerow(["字符", "方格宽度", "左上","右上","右下","左下"])
        # 4. 写入csv文件内容
        for word in pic_data:
            writen_char = word
            writen_position = pic_data[word]
            writen_position1 = writen_position[0]
            writen_position2=writen_position[1]
            writen_position3=writen_position[2]
            writen_position4=writen_position[3]
            csv_writer.writerow([writen_char, h, writen_position1,writen_position2,writen_position3,writen_position4])
        f.close()



        left_up_position = (h,h)
        right_up_position=(h+width,h)
        left_down_position=(h,h+height)
        right_down_position=(h+width,h+height)
        middle_position = (h+int(width/2),h+int(height/2))
        #获取图片的中点的信息

        f = open(file_path + 'keypoint' + str(x) + '.csv', 'w', newline='', encoding='utf-8-sig')
        # 2. 基于文件对象构建 csv写入对象
        csv_writer = csv.writer(f)
        # 3. 构建列表头
        csv_writer.writerow(["图片序号", "左上", "右上", "右下", "左下","图片中点"])
        # 4. 写入csv文件内容
        csv_writer.writerow([x, left_up_position, right_up_position, right_down_position, left_down_position,middle_position])
        f.close()



def standard_data():
    for x in range(amount_start,amount_end):
        img = cv2.imread(file_path + "fushion"+str(x)+".jpg")
        cv2.imwrite(file_path+"result"+str(x)+".jpg",img)
        os.remove(file_path+"fushion"+str(x)+".jpg")
        os.remove(file_path+str(x)+".jpg")
        shutil.copy(file_path+"data"+str(x)+".csv", file_path+"result_data"+str(x)+".csv")
        os.remove(file_path+"data"+str(x)+".csv")






#程序执行的部分
'''
    主要函数功能的介绍
    make_pics()生成图片
    draw_lines()在生成的图片中打上预设的线条
    fill_words()在生成的线条的图片中找到对应的田字格，在填字格内填充相应的随机文字
    make_data()生成规整化数据
'''

if __name__ == '__main__':

    '''基本流程'''
    pic_configs = make_pics()#函数执行完成后，生成了编号从amount_start至amount_end的图片，并返回了所有图片信息的一个列表
    position = draw_lines(pic_configs)#根据make_pics()函数返回的图片信息，进行解析，在每张图片内线条绘制
    #position表示的是所有图片的中的每一行田字格的起始位置
    words_config,words_list=fill_words(position,pic_configs)#在图片中添加文字，函数执行完之后返回文字的相关信息
    fushion_pics(pic_configs)#将生成的田字格图片和白底图片进行融合，使不同大小的田字格最后呈现出相同的图片大小
    make_data(pic_configs,position,words_config,words_list)#该函数用于生成包含图片中文字信息的csv文件
    '''以上函数为必选'''

    '''
    以下函数为3种不同的生成图片的模式
    standard_data()为按照标准模式生成图片，对图片不做任何变换
    rotate_pic()对图片进行随机角度的旋转
    twist_pic()对图片进行随机程度的扭曲
    三种模式只能选择其中一种进行使用，可以设置不同的amount_start，amount_end来生成不同模式的图片
    '''
    #standard_data()  # 删除多余文件，并按照规则重命名
    #rotate_pic()#对图片进行随机角度的旋转
    twist_pic()#对图片进行随机程度的扭曲

    '''该函数为可选函数，可以叠加在以上三种模式中的任何一种上'''
    #make_lights()#给图片添加上光照效果
    '''添加光照涉及大量像素点的计算，速度较慢，需要较长的时间'''


