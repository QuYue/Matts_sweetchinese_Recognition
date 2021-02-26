#%%
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

strs = "往往。，、；！？“”（）：+-=%"
#模板图片
imageFile = "./Pictures/page_0.jpg"
#新文件保存路径
file_save_dir = "./"
 
#初始化参数
x = 0  #横坐标（左右）
y = 0   #纵坐标（上下）
word_size = 60 #文字大小
word_css = './Fonts/kaiti.ttf'
word_css2 = './Fonts/stkaiti.ttf'
word_css3 = './Fonts/simsun.ttc'

# word_css  = "C:\\Windows\\Fonts\\STXINGKA.TTF" #字体文件   行楷
#STXINGKA.TTF华文行楷   simkai.ttf 楷体  SIMLI.TTF隶书  minijianhuangcao.ttf  迷你狂草    kongxincaoti.ttf空心草
 
#设置字体，如果没有，也可以不设置
font = ImageFont.truetype(word_css,word_size)
font2 = ImageFont.truetype(word_css2, word_size)
font3 = ImageFont.truetype(word_css3, word_size)
 
#分割得到数组
im1=Image.open(imageFile) #打开图片
im0 = np.ones([1000, 1000, 3]) * 255
im1 = Image.fromarray(im0.astype('uint8')).convert('RGB')
draw = ImageDraw.Draw(im1)
print(font.getsize(strs))
print(font2.getsize(strs))
print(font3.getsize(strs))

draw.text((x, y),strs,(0,0,0),font=font) #设置位置坐标 文字 颜色 字体
draw.text((x, y+100),strs,(0,0,0),font=font2) #设置位置坐标 文字 颜色 字体
draw.text((x, y+200),strs,(0,0,0),font=font3) #设置位置坐标 文字 颜色 字体

#定义文件名 数字需要用str强转
new_filename = file_save_dir + strs.replace(",","-").replace("\n","-")+".jpg"
# im1.save(new_filename) 
# del draw #删除画笔
# im1.close()  #关闭图片
#%%
punctuation = "。，、；！？“”（）：+-=%" # 标点符号
with open("./fonts/library.txt", "r", encoding='UTF-8') as f:
    chinese = f.readline()
