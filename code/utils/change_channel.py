import os
from tqdm import tqdm
from PIL import Image
path = "../data/train/images/"
def change_channel(path):
    for image in tqdm(os.listdir(path)):
        image_path = os.path.join(path, image)
        img = Image.open(image_path)  # 打开图片
        img = img.convert("RGB")  # 4通道转化为rgb三通道
        save_path = path
        img.save(save_path + image)
def change_channel_mask(path):
    for image in tqdm(os.listdir(path)):
        image_path = os.path.join(path, image)
        img = Image.open(image_path)  # 打开图片
        img = img.convert("L")  # 4通道转化为rgb三通道
        save_path = path
        img.save(save_path + image)


if __name__ == '__main__':
    change_channel_mask("../data/train/masks/")