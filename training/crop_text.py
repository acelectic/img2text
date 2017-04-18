
from PIL import Image
import cv2,os
import numpy as np

dir = r"C:\Users\miniBear\Desktop\nn\img2text\training\new.jpg"
imgname = dir.split('\\')
imgname = imgname[len(imgname)-1].split('.')[0]

print(imgname)


img = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)
w, h = len(img[0]),len(img)




# Read image
im_in = img.copy()
# Threshold.
# Set values equal to or above 220 to 0.
# Set values below 220 to 255.

im_th = cv2.threshold(im_in, 230, 255, cv2.THRESH_BINARY_INV);

tmp_ = []

for i,col in enumerate(img) :
    for j,row in enumerate(col):
        if row == 0:
            tmp_.append([i,j])
            break

# print(tmp)

tmp = [x[0] for x in tmp_]
print(tmp)
print(tmp_)

avg = []
max_dist = 0

for i in range(len(set(tmp))-1):
    x = tmp
    dist = x[i+1]-x[i]
    if dist not in avg:
        avg.append(dist)
    if  dist > max_dist:
        max_dist = dist
avg = [x for x in avg if x > 15]
min = min(avg)
avg_ = sum(set(avg))/len(set(avg))
print(max_dist,avg_)
print(avg,min)

start = tmp[0]
new = []
new.append(start)

for i in range(len(tmp)-1):
    if tmp[i+1] - tmp[i] >= min:
        new.append(tmp[i])
        new.append(tmp[i+1])

print(tmp)
print(new)

print(len(img),len(img[0]))


for i in range(0,len(new)-1,2):
    e = int((((new[i+1]-new[i])*.8)+5)*.65)
    print(e)
    crop = img[new[i]-e:new[i+1]+e , 0:1130]
    newdir = r'C:\Users\miniBear\Desktop\nn\img2text\training\img\\'+imgname
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    cv2.imwrite(newdir+'\\'+imgname+'_'+str(i)+'_.png', crop)
    # cv2.imshow("test",crop)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# # Copy the thresholded image.
# im_floodfill = im_th.copy()
#
# # Mask used to flood filling.
# # Notice the size needs to be 2 pixels than the image.
# h, w = im_th.shape[:2]
# mask = np.zeros((h + 2, w + 2), np.uint8)
#
# # Floodfill from point (0, 0)
# cv2.floodFill(im_floodfill, mask, (0, 0), 255);
#
# # Invert floodfilled image
# im_floodfill_inv = cv2.bitwise_not(im_floodfill)
#
# # Combine the two images to get the foreground.
# im_out = im_th | im_floodfill_inv
#
# # Display images.
# cv2.imshow("Thresholded Image", im_th)
# cv2.imshow("Floodfilled Image", im_floodfill)
# cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
# cv2.imshow("Foreground", im_out)
# cv2.waitKey(0)