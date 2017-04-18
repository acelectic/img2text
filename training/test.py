


from PIL import Image
import cv2,os
import numpy as np


dir = r"C:\Users\miniBear\Desktop\nn\img2text\training\img\new\new_0_.png"
imgname = dir.split('\\')
imgname = imgname[len(imgname)-1].split('.')[0]

print(imgname)


img = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)
w, h = len(img[0]),len(img)


im_th = cv2.threshold(img, 165, 255, cv2.THRESH_BINARY_INV);

img2 = im_th[1]

new = np.rot90(img2,3)

dic = []
siz = []

for i,data in enumerate(new):
    tmp = data.tolist()
    mid = int((len(tmp)/2)+(len(tmp)/2)*.2)
    c = tmp[:mid]

    if 255 not in tmp or 255 not in c:
        # print(i)
        dic.append(i)


for i in range(len(dic)-1):
    tmp = dic[i+1]-dic[i]
    if tmp > 1:
        tt = [dic[i],dic[i+1],tmp]
        siz.append(tt)

print(len(siz))
print(siz)

for i in siz:
    x1 = i[0]-2
    x2 = i[1]+2
    out = img[0:h, x1:x2]

    newdir = r'C:\Users\miniBear\Desktop\nn\img2text\training\img\new\\' + imgname
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    cv2.imwrite(newdir + '\\' + imgname + '_' + str(i) + '.png', out)

    # cv2.imshow("ss", out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()