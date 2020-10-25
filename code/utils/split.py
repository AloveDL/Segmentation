import cv2
import os
import openslide
import numpy as np
import matplotlib.pyplot as plt
import pylab
import numpy as np
from PIL import Image
from openslide import OpenSlide
from tqdm import tqdm
from skimage import io
import xml
import xml.dom.minidom as xmldom
# 需要先设置内存限制，不然仍然会报错内存溢出
Image.MAX_IMAGE_PIXELS = None

level = 2
train_imgs_dir = "../data/train/images"
train_masks_dir = "../data/train/masks"
val_imgs_dir = "../data/val/images"
val_masks_dir = "../data/val/masks"


def train_split():

    index = 0   # patch id
    train_set = []
    val_set = []
    with open("../train_vs_val/slide_train.txt") as tf:
        for file_name in tf.readlines():
            train_set.append(("D:\\ImgDatebase\\"+file_name.split('C/')[1])[:-1])
    with open("../train_vs_val/slide_val.txt") as vf:
        for file_name in vf.readlines():
            val_set.append(("D:\\ImgDatebase\\"+file_name.split('C/')[1])[:-1])

    train_imgs_save_path = "../data/train/images/"
    train_masks_save_path = "../data/train/masks/"
    step = 128  # 步长大小
    for img_id,img_path in tqdm(enumerate(train_set)):
        slide: OpenSlide = openslide.OpenSlide(img_path)
        # print(slide.dimensions)
        mask_path = img_path.replace("Images", "Masks_level2").replace('.tif', '.png').replace('/', '//')
        region = []
        xml_path = img_path.replace("Images", "ConfinedAnnotation").replace('.tif', '.xml').replace('/', '//')
        xmlfilepath = os.path.abspath(xml_path)
        domobj = xmldom.parse(xmlfilepath)
        elementobj = domobj.documentElement
        subElementObj = elementobj.getElementsByTagName("Annotation")

        for node in subElementObj:
            if node.getAttribute('Type') == "Rectangle":
                position = node.getElementsByTagName("Coordinates")

                coordinate = position[0].getElementsByTagName("Coordinate")
                list = []
                for i in coordinate:
                    x1, y1 = i.getAttribute('X'),i.getAttribute('Y')

                # x2,y2 = position[1].childNodes[0].data, position[1].childNodes[1].data
                # x3,y3 = position[2].childNodes[0].data, position[2].childNodes[1].data
                # x4,y4 = position[3].childNodes[0].data, position[3].childNodes[1].data
                    list.append([x1,y1])
                region.append(list)

        mask_img = cv2.imread(mask_path)
        # print(mask_img.shape)
        for roi in region:
            if int(float(roi[0][0])) < 0:
                roi[0][0] = 0
            if int(float(roi[0][1])) < 0:
                roi[0][1] = 0
            if int(float(roi[1][0])) < 0:
                roi[1][0] = 0
            if int(float(roi[2][1])) < 0:
                roi[2][1] = 0
            if int(float(roi[1][1])) < 0:
                roi[1][1] = 0
            X = int(float(roi[0][0]))//4
            Y = int(float(roi[0][1]))//4
            W = (int(float(roi[1][0]))-int(float(roi[0][0])))//4
            H = (int(float(roi[2][1]))-int(float(roi[1][1])))//4
            # print(X,Y,H,W)
            for h in range(Y, Y+H, step):
                for w in range(X, X+W, step):
                    # print(h+128,mask_img.shape[0],w+128,mask_img.shape[1])
                    # if h + 128 > mask_img.shape[0] or w + 128 > mask_img.shape[1]:
                    #     print("X")
                    #     continue
                    tile = np.array(slide.read_region((h, w), level, (128, 128)))
                    # print(tile.shape)
                    # print(tile.shape,mask_img[h:h+128, w:w+128].shape)
                    if tile.mean() >= 230:
                        # print("c")
                        continue
                    # cv2.imshow("d",mask_img[h:h+128, w:w+128])
                    # tile = cv2.imread(tile,1)

                    if tile.shape[0]!=mask_img[h:h+128, w:w+128].shape[0] or tile.shape[1]!=mask_img[h:h+128, w:w+128].shape[1]:
                        print("出现错误")
                        continue
                    plt.imsave('../data/train/masks/'+str(index)+".png", mask_img[h:h+128, w:w+128])
                    # tile = tile.convert("RGB")
                    plt.imsave('../data/train/images/'+str(index)+".png", tile)
                    index += 1
            print("第%d张图片完成制作!" % (img_id+1))


if __name__ == '__main__':
    train_split()
    # pass