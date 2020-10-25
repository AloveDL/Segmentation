import cv2
from PIL import Image
import os
# img = cv2.imread("../data/train/masks/0.png")
# print(img.shape)
#
# import cv2
# from PIL import Image


for i in os.listdir("../data/train/masks/"):
    img = Image.open("../data/train/masks/"+i)
    print(img.size)