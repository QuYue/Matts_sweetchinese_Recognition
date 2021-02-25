#%%
import numpy as np
import cv2

# import numpy as np
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