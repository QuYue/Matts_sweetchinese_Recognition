#%%
import numpy as np
import cv2

#%% 画横线
def drawline(img,pt1,pt2,color,thickness=1,style='dotted1',gap=15):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted0':
        for p in pts:
            cv2.circle(img,p,thickness,color,-1)
    elif style=='dotted1':
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv2.line(img,s,e,color,thickness)
            i+=1
    elif style == 'dotted2':
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            
            s=e
            e=p
            if i%2==1:
                cv2.line(img,s,e,color,thickness)
            if i%2==0:
                cv2.circle(img,(int((s[0]+e[0])/2), int((s[1]+e[1])/2)),thickness,color,-1)
            i+=1
    elif style == 'normal':
        cv2.line(img,pt1,pt2,color,thickness)
        
def drawpoly(img,pts,color,thickness=1,style='dotted1', gap=15):
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        drawline(img,s,e,color,thickness,style, gap=15)

def drawrect(img,pt1,pt2,color,thickness=1,style='dotted1', gap=15):
    pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])] 
    drawpoly(img,pts,color,thickness,style)


#%% 图片叠加
def img_float32(img):
    return img.copy() if img.dtype != 'uint8' else (img/255.).astype('float32')
 
def over(fgimg, bgimg):
    fgimg, bgimg = img_float32(fgimg),img_float32(bgimg)
    (fb,fg,fr,fa),(bb,bg,br,ba) = cv2.split(fgimg),cv2.split(bgimg)
    color_fg, color_bg = cv2.merge((fb,fg,fr)), cv2.merge((bb,bg,br))
    alpha_fg, alpha_bg = np.expand_dims(fa, axis=-1), np.expand_dims(ba, axis=-1)
    
    color_fg[fa==0]=[0,0,0]
    color_bg[ba==0]=[0,0,0]
    
    a = fa + ba * (1-fa)
    a[a==0]=np.NaN
    color_over = (color_fg * alpha_fg + color_bg * alpha_bg * (1-alpha_fg)) / np.expand_dims(a, axis=-1)
    color_over = np.clip(color_over,0,1)
    color_over[a==0] = [0,0,0]
    
    result_float32 = np.append(color_over, np.expand_dims(a, axis=-1), axis = -1)
    return (result_float32*255).astype('uint8')
 
def image_overlay(fgimg, bgimg, xmin = 0, ymin = 0,trans_percent = 1):
    '''
    fgimg: a 4 channel image, use as foreground
    bgimg: a 4 channel image, use as background
    xmin, ymin: a corrdinate in bgimg. from where the fgimg will be put
    trans_percent: transparency of fgimg. [0.0,1.0]
    '''
    #we assume all the input image has 4 channels
    assert(bgimg.shape[-1] == 4 and fgimg.shape[-1] == 4)
    fgimg = fgimg.copy()
    roi = bgimg[ymin:ymin+fgimg.shape[0], xmin:xmin+fgimg.shape[1]].copy()
    
    b,g,r,a = cv2.split(fgimg)
    
    fgimg = cv2.merge((b,g,r,(a*trans_percent).astype(fgimg.dtype)))
    
    roi_over = over(fgimg,roi)
    
    result = bgimg.copy()
    result[ymin:ymin+fgimg.shape[0], xmin:xmin+fgimg.shape[1]] = roi_over
    return result

#%% 光照
#设置图片光照的具体实现
def make_pic_lights(img, light_strength=90):
    # 读取原始图像
    # 获取图像行和列
    rows, cols = img.shape[:2]
    image = img.astype(np.int64)

    '''可修改'''
    centerX = np.random.randint(0,cols)
    centerY = np.random.randint(0,rows)# 设置中心点

    radius = np.random.randint(min(rows, cols)/3, max(rows, cols)/2)

    '''可修改'''
    strength = np.random.randint(0,light_strength)# 设置光照强度

    # 图像光照特效
    distance = np.ones(image.shape[:2])
    for i in range(rows):
        for j in range(cols):
            distance[i,j] = (centerY - i)**2 + (centerX - j)**2 
    result = np.round(strength * (1.0 - np.sqrt(distance) / radius))[..., np.newaxis].astype(np.int64)
    image[...,:3] += result
    image[image<0] = 0
    image[image>255] = 255
    image = image.astype(np.uint8)
    return image

#%%
if __name__ == '__main__':
    im = np.zeros((800,800,3),dtype='uint8')
    s = np.array([234,222])
    e = np.array([500,700])
    drawline(im, tuple(s), tuple(e),(0,255,255),1,'dotted0', gap=25)
    drawline(im, tuple(s+50), tuple(e+50),(0,255,255),1,'dotted1', gap=25)
    drawline(im, tuple(s+100), tuple(e+100),(0,255,255),1,'dotted2', gap=25)

    drawrect(im, tuple(s), tuple(e),(0,255,255),1,'dotted0', gap=25)
    drawrect(im, tuple(s+50), tuple(e+50),(0,255,255),1,'dotted1', gap=25)
    drawrect(im, tuple(s+100), tuple(e+100),(0,255,255),1,'dotted2', gap=25)

    cv2.imshow('im',im)
    cv2.waitKey()

