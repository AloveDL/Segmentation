from utils.datasets import *
from utils.tools import *
from torch.utils.data import DataLoader
import cv2
import openslide
import matplotlib.pyplot as plt
from utils.change_channel import *
import numpy as np
from PIL import Image
from openslide import OpenSlide
from tqdm import tqdm
import xml.dom.minidom as xmldom

def eval_net(net, loader ,device):

    net.eval()
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    tot = 0
    map = 0
    jaccard = 0
    recall = 0
    precision = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)



            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            tot += dice_coeff(pred, true_masks).item()
            map += pixel_accurary(pred, true_masks).item()
            jaccard += get_jaccard(pred, true_masks).item()
            recall += get_recall(pred, true_masks).item()
            precision += get_precision(pred, true_masks).item()
            pbar.update()


    net.train()
        # dice_list.append(tot / n_val)
        # pixel_list.append(map / n_val)
        # jac_list.append(jaccard / n_val)
        # r_list.append(recall / n_val)
        # p_list.append(precision/n_val)
    return tot / n_val, map / n_val, jaccard / n_val,recall / n_val,precision/n_val
def remove():
    img_path = "data/val/images"
    mask_path = "data/val/masks"
    for i in os.listdir(img_path):
        os.remove(i)
    for j in os.listdir(mask_path):
        os.remove(j)